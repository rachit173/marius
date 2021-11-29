#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"
#include "communication.h"

#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <exception>
#include <filesystem>


#include <torch/torch.h>
#include <c10d/FileStore.hpp>
#include <c10d/PrefixStore.hpp>
#include <c10d/Store.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/GlooDeviceFactory.hpp>
#include <c10d/frontend.hpp>

typedef std::shared_mutex RwLock;
typedef std::unique_lock<RwLock> WriteLock;
typedef std::shared_lock<RwLock> ReadLock;
namespace fs = std::filesystem;


///



//
// TODO List:
// 1. Process group (pg_) is not thread safe, requires locking.
// 2. Multiple process groups are needed, atleast one for 
// metadata communication and one for bulk transfer.
// 3. Interactions need to be transferred to coordinator.
class WorkerNode {
  public:
    explicit WorkerNode(
      std::shared_ptr<c10d::ProcessGroupGloo> pg,
      int rank,
      int capacity,
      int num_partitions,
      int num_workers
    ):
    pg_(pg), 
    rank_(rank),
    capacity_(capacity),
    num_partitions_(num_partitions),
    num_workers_(num_workers),
    tag_generator_(rank, num_workers) {
      coordinator_rank_ = num_workers_; 
      sleep_time_ = 500; // us
      int partition_size_ = 10;
      int embedding_dims_ = 12;
      processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      trained_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
    }
    void start_working(DataSet* trainset, DataSet* evalset, 
                        Trainer* trainer, Evaluator* evaluator, 
                        PartitionBufferStorage* embeds,             // Require the storage for node embeddings to be PartitionBufferStorage
                        PartitionBufferStorage* embeds_state) {
      {
        // Set the private parameters.
        trainset_ = trainset;  // Dataset containing the batches queue
        evalset_ = evalset;
        trainer_ = trainer;     // Trainer container 
        evaluator_ = evaluator;
        embeds_ = embeds;
        embeds_state_ = embeds_state;
        pb_embeds_ = embeds_->getPartitionBuffer();
        pb_embeds_state_ = embeds_state_->getPartitionBuffer();
        pipeline_ = ((PipelineTrainer*)trainer_)->getPipeline();

        // Allocate memory for send and receive buffers
        // TODO(scaling): Currently it seems a larger partition is being allocation than needed.
        partition_size_ = pb_embeds_->getPartitionSize();
        std::cout << "Partition size of send buffer : " << partition_size_;
        dtype_size_ = pb_embeds_->getDtypeSize();
        embedding_size_ = pb_embeds_->getEmbeddingSize();
        if(posix_memalign(&send_buffer_, 4096, partition_size_ * embedding_size_ * dtype_size_)){
          SPDLOG_ERROR("Error in allocating memory for send buffer");
          exit(1);
        }
        // TODO(scaling): Receive buffer.
      }
      // Need 
      // 1. Request partitions when below capacity.
      // 2. Mark partitions when training done and dispatch partition metadata to co-ordinator
      // 3. Transfer partitions to other workers
      // 4. Run actual training

      threads_.emplace_back(std::thread([&](){
        this->RunTrainer();
      }));

      // Wait for initializing buffers
      while(!pb_embeds_->getLoaded()){
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      while (!pb_embeds_state_->getLoaded()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      // while(!pb_embeds_->getLoaded() && !pb_embeds_state_->getLoaded() ) );
      threads_.emplace_back(std::thread([&](){
        this->RequestPartitions();
      }));
      threads_.emplace_back(std::thread([&](){
        this->ProcessPartitions();
      }));
      
      threads_.emplace_back(std::thread([&](){
        this->TransferPartitionsToWorkerNodes();
      }));

      for (auto& t: threads_) {
        t.join();
      }
    }
    void stop_working() {}
  private:
    void ClearProcessedInteractions() {
      assert(processed_interactions_.size() == num_partitions_);
      assert(processed_interactions_[0].size() == num_partitions_);
      for (int i = 0; i < num_partitions_; i++) {
        for (int j = 0; j < num_partitions_; j++) {
          processed_interactions_[i][j] = 0;
        }
      }
    }
    
    // TODO(scaling): Move parts to PartitionMetadata
    PartitionMetadata receivePartition(int srcRank) {
      torch::Tensor tensor = torch::zeros({1}) + 1; // command is 1 for request partition.
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, srcRank, tag_generator_.getCoordinatorCommandTag());
      if (send_work) {
        send_work->wait();
      }
      std::cout << "Started Receiving" << std::endl;
      // Receive the returned partition
      torch::Tensor part_tensor = torch::zeros({num_partitions_+3});
      std::vector<at::Tensor> part_tensor_vec({part_tensor});
      auto recv_work = pg_->recv(part_tensor_vec, srcRank, tag_generator_.getCoordinatorSpecificCommunicationTag());
      if (recv_work) {
        recv_work->wait();
      }
      return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
    }

    int getSize(){
      ReadLock r_lock(avail_parts_rw_mutex_);
      int size = avail_parts_.size();
      r_lock.unlock();
      return size;
    }

    void RequestPartitions() {
      while (1) {
        int size = getSize();
        // SPDLOG_INFO("Number of elements in available partitions: {}", size);
        while (size < capacity_) {
          std::cout << size << " " << capacity_ << std::endl;
          PartitionMetadata p = receivePartition(coordinator_rank_);
          SPDLOG_INFO("Received partition metadata from co-ordinator: {} {}", p.idx, p.src);
          
          // Partition not available
          if(p.idx == -1 && p.src == -1){
            SPDLOG_INFO("Partition not available... Sleeping..");
            break;
          }
          
          TransferPartitionsFromWorkerNodes(p);
          ProcessNewPartition(p);
          size = getSize();
        }
        // Increase the sleep time to reduce lock contention for avail_parts_rw_mutex_
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_)); // TODO(rrt): reduce this later;
        // if complete --> go wait on some trigger
      }
    }

    // TODO(scaling): Move to PartitionMetadata
    bool sendPartition(PartitionMetadata part, int dstRank) {
      torch::Tensor tensor = part.ConvertToTensor();
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, dstRank, tag_generator_.getCoordinatorSpecificCommunicationTag());
      if (send_work) {
        send_work->wait();
      } else {
        return false;
      }
      return true;
    }
    
    void DispatchPartitionsToCoordinator(PartitionMetadata part) {
      // TODO: Put into eviction queue which belongs to partition buffer
      // TODO: PartitionMetadata -> interactions ==> processed_interactions_row for index part.idx
      for(int i = 0; i < num_partitions_; i++){
        part.interactions[i] = part.interactions[i] | processed_interactions_[part.idx][i];
      }
      torch::Tensor tensor = torch::zeros({1}) + 2; // command is 2 for dispatch partition.
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, coordinator_rank_, tag_generator_.getCoordinatorCommandTag());
      if (send_work) {
        send_work->wait();
      }
      part.src = rank_;
      SPDLOG_INFO("Dispatching partition {}", part.idx);

        pb_embeds_->addPartitionForEviction(part.idx);
        pb_embeds_state_->addPartitionForEviction(part.idx);
      
      sendPartition(part, coordinator_rank_);
      SPDLOG_INFO("Dispatched partition {}", part.idx);
    }

    void TransferPartitionsFromWorkerNodes(PartitionMetadata part) {
      if (part.src == -1) {
        // Already have the partition.
      } else if (part.src == rank_) {
        // Already have the partition.
      } else {
        // ask the other worker for the partition
        torch::Tensor request = torch::zeros({1}) + part.idx;
        std::vector<at::Tensor> request_tensors({request});
        auto send_work = pg_->send(request_tensors, part.src, tag_generator_.getTagWhenRequesterCommandPath(part.src));
        if (send_work) {
          send_work->wait();
        }
        // Receive metadata
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        torch::Tensor node_embed_tensor = torch::zeros({1, 5}, options);
        std::vector<torch::Tensor> node_embed_tensors({node_embed_tensor});
        auto recv_work = pg_->recv(node_embed_tensors, part.src, tag_generator_.getTagWhenRequesterDataPath(part.src));
        if (recv_work) {
          recv_work->wait();
        }
        // Create Partition from metadata received
        auto partition = Partition::ConvertToPartition(node_embed_tensors[0]);

        // Process: 
        // 1. Allocate memory for the receiving the incoming partition (posix_memalign)
        // 2. Receive data into tensor, which points under the hood to the same allocated memory
        // 3. Write the recvd partition to disk(partition file)
        
        // Receive partition data
        options = torch::TensorOptions().dtype(partition->dtype_);
        // 1. Allocate memory
        if(posix_memalign(&(partition->data_ptr_), 4096, partition->partition_size_ * partition->embedding_size_ * partition->dtype_size_)){
          SPDLOG_ERROR("Error in allocating memory to receive data");
          exit(1);
        }
        torch::Tensor tensor_data_recvd = torch::from_blob(partition->data_ptr_, {partition->partition_size_,partition->embedding_size_},partition->dtype_);              
        // TODO: [Optimization]: class Single large space, any size of partition can be copied there
        // then directly use pwrite to copy to correct portion of node embeddings                                                   
        partition->tensor_ = tensor_data_recvd;
        
        // 2. Receive the tensor having data from the worker.
        std::vector<torch::Tensor> tensors_data_recvd({tensor_data_recvd});
        auto recv_data_work = pg_->recv(tensors_data_recvd, part.src, tag_generator_.getTagWhenRequesterDataPath(part.src));
        if (recv_data_work) {
          recv_data_work->wait();
        }

        // 3. Write fetched partition to partitioned file.
        PartitionBuffer *partition_buffer = pb_embeds_;
        std::vector<Partition *>& partition_table = partition_buffer->getPartitionTable();
        PartitionedFile *partition_file = partition_buffer->getPartitionedFile();
        partition_file->writePartition(partition.get());

        //4. TODO: Receive optimizer state partition and write it to its own partitioned file
      }

      {
        forceToBuffer(pb_embeds_, part.idx);
        forceToBuffer(pb_embeds_state_, part.idx);
      }

      // Add the received partitions to avail_parts_ vector.
      {
        WriteLock w_lock(avail_parts_rw_mutex_);
        SPDLOG_INFO("Pushed to avail parts: {}", part.idx);
        avail_parts_.push_back(part);
      }
    }

    void forceToBuffer(PartitionBuffer *partition_buffer, int partition_idx){
      // Admit partition into partition buffer forcefully
      Partition *partition = partition_buffer->getPartitionTable()[partition_idx];
      partition_buffer->admitWithLock(partition);
    }

    void ProcessNewPartition(PartitionMetadata p) {
      // Merge local view of partition with global view fetched from co-ordinator
      for(int i = 0; i < num_partitions_; i++){
        processed_interactions_[p.idx][i] = p.interactions[i] | processed_interactions_[p.idx][i];
        trained_interactions_[p.idx][i] = p.interactions[i] | trained_interactions_[p.idx][i];
      }
      // Acquire lock on avail parts and add new batches to be processed to the dataset queue.
      std::vector<pair<int, int>> interactions;
      {
        ReadLock r_lock(avail_parts_rw_mutex_);
        for (const auto &pj : avail_parts_) {
          if (pj.idx == p.idx) {
            if (!processed_interactions_[p.idx][pj.idx]) {
              processed_interactions_[p.idx][pj.idx] = 1;
              interactions.push_back({p.idx, pj.idx});
            }
          } else {
            assert(pj.idx != p.idx);
            if (!processed_interactions_[p.idx][pj.idx]) {
              interactions.push_back({p.idx, pj.idx});
              processed_interactions_[p.idx][pj.idx] = 1;
            }
            if (!processed_interactions_[pj.idx][p.idx]) {
              interactions.push_back({pj.idx, p.idx});
              processed_interactions_[pj.idx][p.idx] = 1;
            }
          }
        }
      }
      if (interactions.size() > 0) { 
        SPDLOG_INFO("Generated {} interactions", interactions.size());
      }
      for (auto interaction: interactions) {
        int src = interaction.first;
        int dst = interaction.second;
        // Add batch to dataset batches queue. 
        trainset_->addBatchScaling(src, dst);
        SPDLOG_INFO("Pushed ({}, {}) to dataset queue", src, dst);
      }
    }
    
    void TransferPartitionsToWorkerNodes() {
      // Receive the request and transfer the partition from the node map
      while (1) {
        // Receive request for transfer
        torch::Tensor request = torch::zeros({1});
        std::vector<at::Tensor> tensors({request});
        auto recv_work = pg_->recvAnysource(tensors, tag_generator_.getTagWhenReceiverCommandPath());
        int srcRank;
        if (recv_work) {
          recv_work->wait();
          srcRank = recv_work->sourceRank();
        }
        // send partition metadata
        int part_idx = tensors[0].data_ptr<float>()[0];
        PartitionBuffer* partition_buffer = pb_embeds_;
		  	std::vector<Partition*>& partition_table = partition_buffer->getPartitionTable();
        PartitionedFile* partition_file = partition_buffer->getPartitionedFile();
				torch::Tensor partition_metadata = partition_table[part_idx]->ConvertMetaDataToTensor();
				std::vector<torch::Tensor> tensors_to_send({partition_metadata});
        // send metadata
        auto send_serialized_partition = pg_->send(tensors_to_send, srcRank, tag_generator_.getTagWhenReceiverDataPath(srcRank));
        if (send_serialized_partition) {
          send_serialized_partition->wait();
        }                
        // send partition data
        // 1. Read partition from disk
        // 2. Convert to tensor and send
        partition_file->readPartition(send_buffer_, partition_table[part_idx]);
        // Can be dispatched to co-ordinator but still be present in the buffer till not actually evicted
        // assert(!partition_table[part_idx]->present_);

        torch::Tensor tensor_data_to_send = partition_table[part_idx]->ConvertDataToTensor();
        std::vector<torch::Tensor> tensors_data_to_send({tensor_data_to_send});
        // send partition data
        auto send_part_data = pg_->send(tensors_data_to_send, srcRank, tag_generator_.getTagWhenReceiverDataPath(srcRank));
        if (send_part_data) {
          send_part_data->wait();
        }
      }
    }
    
    void ProcessPartitions() {
      // Generate interactions to be processed.
      // @TODO(scaling): Implement optimal strategies.
      // The strategy computation is expected to be fast and thus we can hold locks
      while (1) {
        std::vector<PartitionMetadata> partitions_done;
        {
          // Lock contention possible as acquired every iteration by this function as well as RequestPartitions
          WriteLock w_lock(avail_parts_rw_mutex_);
          const int num_batches_processed = pipeline_->getCompletedBatchesSize();
          const int avail_parts_size = avail_parts_.size();
          SPDLOG_INFO("Size of available partitions: {}", avail_parts_size);
          // check if completed --> and set pipeline completedScaling to true

          if (avail_parts_size == capacity_) {
            for (int i = 0; i < num_batches_processed; i++) {
              PartitionBatch* batch = (PartitionBatch*)pipeline_->completed_batches_[i];
              int src_idx = batch->src_partition_idx_;
              int dst_idx = batch->dst_partition_idx_;
              if (trained_interactions_[src_idx][dst_idx] == 0) {
                trained_interactions_[src_idx][dst_idx] = 1;
                std::cout << "Trained on partition: (" << src_idx << "," << dst_idx << ")" << std::endl;
              }
            }
            std::vector<PartitionMetadata> avail_parts_replacement;
            for (int i = 0; i < avail_parts_size; i++) {
              bool done = true;
              const auto& pi = avail_parts_[i];
              for (int j = 0; j < avail_parts_size; j++) {
                const auto& pj = avail_parts_[j];
                if (!trained_interactions_[pi.idx][pj.idx]) { done = false; break; }
                if (!trained_interactions_[pj.idx][pi.idx]) { done = false; break; }
              }
              if (done && partitions_done.empty()) {
                partitions_done.push_back(pi);
              } else {
                avail_parts_replacement.push_back(pi);
              }
            }
            // For debugging
            if(avail_parts_.size() != avail_parts_replacement.size()){
              SPDLOG_INFO("Available parts changed...");
              std::cout << "Old available parts:" << std::endl;
              for(int i = 0; i < avail_parts_.size(); i++)std::cout << avail_parts_[i].idx << ", ";
              std::cout << std::endl;

              std::cout << "New available parts:" << std::endl;
              for (int i = 0; i < avail_parts_replacement.size(); i++)
                std::cout << avail_parts_replacement[i].idx << ", ";
              std::cout << std::endl;
            }
            // Update avail_parts.
            avail_parts_ = avail_parts_replacement;
          }
        }

        // TODO: Introduce a queue and put this into separate thread
        for (auto p : partitions_done) {
          // queue.push(p)
          SPDLOG_INFO("Dispatching partition {} to co-ordinator..", p.idx);
          DispatchPartitionsToCoordinator(p);
        }
        // sleep
        // Increase sleep time to reduce contention for lock
        // TODO(rrt): Replace this by a condition variable.
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
      }
    }
    
    void RunTrainer() {
      int num_epochs = 1;
      trainer_->train(num_epochs);
      // TODO(scaling): Add evaluation code.
    }
  private:
  std::shared_ptr<c10d::ProcessGroupGloo> pg_;
  int rank_;
  int num_partitions_;
  int num_workers_;
  int capacity_;
  int coordinator_rank_;
  int sleep_time_;
  void* send_buffer_;
  void* receive_buffer_;
  mutable std::shared_mutex avail_parts_rw_mutex_;
  std::vector<PartitionMetadata> avail_parts_;
  mutable std::shared_mutex dispatch_parts_rw_mutex_;
  std::queue<PartitionMetadata> dispatch_parts_;
  mutable std::shared_mutex transfer_receive_parts_rw_mutex_;
  std::queue<PartitionMetadata> transfer_receive_parts_;
  std::vector<std::thread> threads_;
  mutable std::shared_mutex node_map_rw_mutex_;
  std::vector<std::shared_ptr<torch::Tensor>> node_map;
  vector<vector<int>> processed_interactions_;
  vector<vector<int>> trained_interactions_;
  int partition_size_;
  int dtype_size_;
  int embedding_size_; 
  int embedding_dims_;
  DataSet* trainset_;
  DataSet* evalset_;
  Trainer* trainer_;
  Evaluator* evaluator_;
  Pipeline* pipeline_;
  PartitionBufferStorage* embeds_;
  PartitionBufferStorage* embeds_state_;
  PartitionBuffer* pb_embeds_;
  PartitionBuffer* pb_embeds_state_;
  WorkerTagGenerator tag_generator_;
};


int main(int argc, char* argv[]) {
  marius_options = parseConfig(argc, argv); // marius options is an extern variable form config.h that is globally used across the library.
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
  string base_dir = "/proj/uwmadison744-f21-PG0/groups/g007";
  auto filestore = c10::make_intrusive<c10d::FileStore>(base_dir + "/rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);
  // auto dev = c10d::GlooDeviceFactory::makeDeviceForInterface("lo");
  std::chrono::milliseconds timeout(10000000);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("eth1"));
  options->timeout = timeout;
  options->threads = options->devices.size() * 2;
  auto pg = std::make_shared<c10d::ProcessGroupGloo>(
    prefixstore, rank, world_size, options);
  int num_partitions = marius_options.storage.num_partitions;
  int capacity = marius_options.storage.buffer_capacity;
  WorkerNode worker(pg, rank, capacity, num_partitions, world_size-1);
  std::string log_file = marius_options.general.experiment_name;
  MariusLogger marius_logger = MariusLogger(log_file);
  spdlog::set_default_logger(marius_logger.main_logger_);
  marius_logger.setConsoleLogLevel(marius_options.reporting.log_level);
  bool gpu = false;
  if (marius_options.general.device == torch::kCUDA) {
      gpu = true;
  }
  Timer preprocessing_timer = Timer(gpu);
  preprocessing_timer.start();
  SPDLOG_INFO("Start preprocessing");

  DataSet *train_set;
  DataSet *eval_set;

  Model *model = initializeModel(marius_options.model.encoder_model, marius_options.model.decoder_model);

  tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> storage_ptrs = initializeTrain();
  Storage *train_edges = get<0>(storage_ptrs);
  Storage *eval_edges = get<1>(storage_ptrs);
  Storage *test_edges = get<2>(storage_ptrs);

  Storage *embeds = get<3>(storage_ptrs);
  Storage *embeds_state = get<4>(storage_ptrs);

  Storage *src_rel = get<5>(storage_ptrs);
  Storage *src_rel_state = get<6>(storage_ptrs);
  Storage *dst_rel = get<7>(storage_ptrs);
  Storage *dst_rel_state = get<8>(storage_ptrs);

  bool will_evaluate = !(marius_options.path.validation_edges.empty() && marius_options.path.test_edges.empty());

  train_set = new DataSet(train_edges, embeds, embeds_state, src_rel, src_rel_state, dst_rel, dst_rel_state);
  SPDLOG_INFO("Training set initialized");
  if (will_evaluate) {
      eval_set = new DataSet(train_edges, eval_edges, test_edges, embeds, src_rel, dst_rel);
      SPDLOG_INFO("Evaluation set initialized");
  }

  preprocessing_timer.stop();
  int64_t preprocessing_time = preprocessing_timer.getDuration();

  SPDLOG_INFO("Preprocessing Complete: {}s", (double) preprocessing_time / 1000);

  Trainer *trainer;
  Evaluator *evaluator;

  if (marius_options.training.synchronous) {
      trainer = new SynchronousTrainer(train_set, model);
  } else {
      trainer = new PipelineTrainer(train_set, model);
  }

  if (will_evaluate) {
      if (marius_options.evaluation.synchronous) {
          evaluator = new SynchronousEvaluator(eval_set, model);
      } else {
          evaluator = new PipelineEvaluator(eval_set, model);
      }
  }
  // train_set, eval_set, trainer, evaluator are not populated.
  // Sending them to worker. 
  worker.start_working(train_set, eval_set, 
                        trainer, evaluator, 
                        (PartitionBufferStorage*)embeds, 
                        (PartitionBufferStorage*)embeds_state);
  worker.stop_working();
  embeds->unload(true);
  src_rel->unload(true);
  dst_rel->unload(true);


  // garbage collect
  delete trainer;
  delete train_set;
  if (will_evaluate) {
      delete evaluator;
      delete eval_set;
  }

  freeTrainStorage(train_edges, eval_edges, test_edges, embeds, embeds_state, src_rel, src_rel_state, dst_rel, dst_rel_state);

}

/*TODO:
1. Correctness (accuracy)
2. Terminating condition
  -- Signal to the pipeline that epoch is done : variable in pipeline set by worker based on interactions matrix
  -- Clear epoch specific data structures:
      1. processed_interactions_ , 2. trained_interactions_, 3. eviction queues
  -- Blocking pop 
3. Reduce interactions by a better policy
  -- coordinator as central unit
4. Handling symmetric interactions in co-ordinator
5. Optimizer state fetching(to and from)
6. Multi node communication
7. Run in twitter data to observe bottlenecks
8. Prefetching fix
*/