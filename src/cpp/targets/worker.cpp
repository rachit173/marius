#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"

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
    num_workers_(num_workers) {
      coordinator_rank_ = num_workers_; 
      sleep_time_ = 100; // us
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
      }
      // Need 
      // 1. Request partitions when below capacity.
      // 2. Dispatch partitions
      // 3. Transfer partitions to other workers
      threads_.emplace_back(std::thread([&](){
        this->RequestPartitions();
      }));
      threads_.emplace_back(std::thread([&](){
        this->DispatchPartitionsToCoordinator();
      }));
      threads_.emplace_back(std::thread([&](){
        this->TransferPartitionsFromWorkerNodes();
      }));
      threads_.emplace_back(std::thread([&](){
        this->TransferPartitionsToWorkerNodes();
      }));
      threads_.emplace_back(std::thread([&](){
        this->ProcessPartitions();
      }));
      threads_.emplace_back(std::thread([&](){
        this->RunTrainer();
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
      auto send_work = pg_->send(tensors, srcRank, 0);
      if (send_work) {
        send_work->wait();
      }
      std::cout << "Started Receiving" << std::endl;
      // Receive the returned partition
      torch::Tensor part_tensor = torch::zeros({num_partitions_+3});
      std::vector<at::Tensor> part_tensor_vec({part_tensor});
      auto recv_work = pg_->recv(part_tensor_vec, srcRank, 0);
      if (recv_work) {
        recv_work->wait();
      }
      return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
    }
    void RequestPartitions() {
      while (1) {
        // if ()
        int size;
        {
          ReadLock r_lock(avail_parts_rw_mutex_);
          size = avail_parts_.size() + transfer_receive_parts_.size();
        }
        while (size < capacity_) {
          std::cout << size << " " << capacity_ << std::endl;
          PartitionMetadata p = receivePartition(coordinator_rank_);
          std::cout << "Received partition: " << p.idx << " " << p.src << std::endl;
          {
            WriteLock w_lock(transfer_receive_parts_rw_mutex_);
            transfer_receive_parts_.push(p);
          }
          size++;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
      }
    }
    // TODO(scaling): Move to PartitionMetadata
    bool sendPartition(PartitionMetadata part, int dstRank) {
      torch::Tensor tensor = part.ConvertToTensor();
      std::cout << "tensor to send " << tensor << std::endl;
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, dstRank, 0);
      if (send_work) {
        send_work->wait();
      } else {
        return false;
      }
      return true;
    }
    void DispatchPartitionsToCoordinator() {
      while (1) {
        while (1) {
          PartitionMetadata part(-1, -1, num_partitions_);
          {
            WriteLock w_lock(dispatch_parts_rw_mutex_);
            if (dispatch_parts_.empty()) break;
            part = dispatch_parts_.front();
            dispatch_parts_.pop();
          }
          torch::Tensor tensor = torch::zeros({1}) + 2; // command is 2 for dispatch partition.
          std::vector<at::Tensor> tensors({tensor});
          auto send_work = pg_->send(tensors, coordinator_rank_, 0);
          if (send_work) {
            send_work->wait();
          }
          part.src = rank_;
          sendPartition(part, coordinator_rank_);      
        }
        // TODO: all sleep based code can be converted to conditional
        // variable.
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
      }
    }
    void fetchFromStorage(int part_num) {}
    void TransferPartitionsFromWorkerNodes() {
      while (1) {
        {
          while (1) {
            PartitionMetadata part(-1, -1, num_partitions_);
            {
              WriteLock w_lock(transfer_receive_parts_rw_mutex_);
              if (transfer_receive_parts_.empty()) break;
              part  = transfer_receive_parts_.front();
              transfer_receive_parts_.pop();
            }
            std::cout << "Getting partition " << part.idx << " from " << part.src << std::endl; 
            if (part.src == -1) {
              // Already have the partition.
              fetchFromStorage(part.idx);
            } else if (part.src == rank_) {
              // Already have the partition.
            } else {
              int tag = 1;
              // ask the other worker for the partition
              torch::Tensor request = torch::zeros({1}) + part.idx;
              std::vector<at::Tensor> request_tensors({request});
              auto send_work = pg_->send(request_tensors, part.src, tag);
              if (send_work) {
                send_work->wait();
              }
              // Receive the tensor from the worker.
              std::vector<at::Tensor> node_embed_tensors({torch::zeros({partition_size_, embedding_dims_})});
              auto recv_work = pg_->recv(node_embed_tensors, part.src, tag);
              if (recv_work) {
                recv_work->wait();
              }
              // Set the received tensor to the node map
              node_map[part.idx] = std::make_shared<torch::Tensor>(std::move(node_embed_tensors[0]));
            }
            // Add the received partitions to avail_parts_ vector.
            {
              WriteLock w_lock(avail_parts_rw_mutex_);
              std::cout << "Pushed to avail parts: " << part.idx << std::endl;
              avail_parts_.push_back(part);
            }
          }
          // TODO(rrt): convert this to conditional code.
          std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
        }
        
      }
    }
    void TransferPartitionsToWorkerNodes() {
      // Receive the request and transfer the partition from the node map
      while (1) {
        int tag = 1;
        // Receive request for transfer
        torch::Tensor request = torch::zeros({1});
        std::vector<at::Tensor> tensors({request});
        auto recv_work = pg_->recvAnysource(tensors, tag);
        int srcRank;
        if (recv_work) {
          recv_work->wait();
          srcRank = recv_work->sourceRank();
        }
        int part_idx = tensors[0].data_ptr<float>()[0];
        // send partition
        // std::vector<at::Tensor> tensors;
        // tensors.push_back(std::move(*node_map[part_idx]));
        // pg_->send();



        // empty that entry in the node map after sending
        node_map[part_idx] = nullptr;
      }
    }
    void ProcessPartitions() {
      // Generate interactions to be processed.
      // @TODO(scaling): Implement optimal strategies.
      // The strategy computation is expected to be fast and thus we can hold locks
      while (1) {
        std::vector<pair<int, int>> interactions;
        std::vector<PartitionMetadata> partitions_done;
        {
          WriteLock r_lock(avail_parts_rw_mutex_);
          for (const auto& pi : avail_parts_) {
            for (const auto& pj : avail_parts_) {
              if (processed_interactions_[pi.idx][pj.idx] == 0) {
                interactions.push_back({pi.idx, pj.idx});
                processed_interactions_[pi.idx][pj.idx] = 1;
              }
            }
          }
          int num_batches_processed = pipeline_->getCompletedBatchesSize();
          if (avail_parts_.size() == capacity_) {
            for (int i = 0; i < num_batches_processed; i++) {
              PartitionBatch* batch = (PartitionBatch*)pipeline_->completed_batches_[i];
              int src_idx = batch->src_partition_idx_;
              int dst_idx = batch->dst_partition_idx_;
              if (trained_interactions_[src_idx][dst_idx] == 0) {
                trained_interactions_[src_idx][dst_idx] = 1;
                std::cout << "Trained on partition: (" << src_idx << "," << dst_idx << ")" << std::endl;
              }
            }
          }
          std::vector<PartitionMetadata> avail_parts_replacement;
          for (const auto& pi : avail_parts_) {
            bool done = true;
            for (const auto& pj : avail_parts_) {
              if (!trained_interactions_[pi.idx][pj.idx]) { done = false; break; }
            }
            if (done) {
              partitions_done.push_back(pi);
            } else {
              avail_parts_replacement.push_back(pi);
            }
          }
          avail_parts_ = avail_parts_replacement;
        }
        for (auto p : partitions_done) {
          WriteLock w_lock(dispatch_parts_rw_mutex_);
          dispatch_parts_.push(p);
        }
        if (interactions.size() > 0) std::cout << "Generated Interaction vector " << interactions.size() << std::endl;
        for (auto interaction: interactions) {
          int src = interaction.first;
          int dst = interaction.second;
          // Add batch to dataset batches queue. 
          trainset_->addBatchScaling(src, dst);
          std::cout << "Pushed (" << src << ", " << dst << ") to dataset queue" << std::endl;
        }
        // sleep
        // TODO(rrt): Replace this by a condition variable.
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_));
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
};


int main(int argc, char* argv[]) {
  marius_options = parseConfig(argc, argv); // marius options is an extern variable form config.h that is globally used across the library.
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
  auto filestore = c10::make_intrusive<c10d::FileStore>("./rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);
  // auto dev = c10d::GlooDeviceFactory::makeDeviceForInterface("lo");
  std::chrono::milliseconds timeout(100000);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("lo"));
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