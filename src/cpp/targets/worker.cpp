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
#include <algorithm>


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
      int num_workers,
      int num_epochs,
      bool gpu
    ):
    pg_(pg), 
    rank_(rank),
    capacity_(capacity),
    num_partitions_(num_partitions),
    num_workers_(num_workers),
    tag_generator_(rank, num_workers),
    num_epochs_(num_epochs),
    timestamp_(0),
    gpu_(gpu) {
      coordinator_rank_ = num_workers_; 
      sleep_time_ = 500; // us
      int partition_size_ = 10;
      int embedding_dims_ = 12;
      processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      trained_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      perf_metrics_label_ = "[Performance Metrics]";
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
        SPDLOG_TRACE("Partition size of send buffer : {}", partition_size_);
        dtype_size_ = pb_embeds_->getDtypeSize();
        embedding_size_ = pb_embeds_->getEmbeddingSize();
        if(posix_memalign(&send_buffer_, 4096, partition_size_ * embedding_size_ * dtype_size_)){
          SPDLOG_ERROR("Error in allocating memory for send buffer");
          exit(1);
        }
        if (posix_memalign(&receive_buffer_, 4096, partition_size_ * embedding_size_ * dtype_size_)) {
          SPDLOG_ERROR("Error in allocating memory for receive buffer");
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

      threads_.emplace_back(std::thread([&](){
        this->PrepareForNextEpoch();
      }));
      // Evaluator thread is run only for worker
      // with rank 0.
      if (rank_ == 0) {
        threads_.emplace_back(std::thread([&]() {
          this->ServiceEvaluationRequest();
        }));
      }
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
      SPDLOG_TRACE("Started Receiving");
      // Receive the returned partition
      torch::Tensor part_tensor = torch::zeros({num_partitions_+kPartititionMetadataSerde});
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
      while (timestamp_ < num_epochs_) {
        int size = getSize();
        SPDLOG_TRACE("Number of elements in available partitions: {}", size);
        while (size < capacity_) {
          SPDLOG_TRACE("Avail parts --> size: {}, capacity: {}", size, capacity_);

          PartitionMetadata p = receivePartition(coordinator_rank_);
          SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Received Partition Location Info\", 'data':('partition index':{}, 'src':{}, 'timestamp':{}))", perf_metrics_label_, timestamp_ + 1, rank_, timestamp_ + 1, rank_, p.idx, p.src, p.timestamp);
          SPDLOG_TRACE("Received partition metadata from co-ordinator: Index: {}, source: {}, timestamp: {}", p.idx, p.src, p.timestamp);
          
          // Partition not available
          if(p.idx == -1 && p.src == -1){
            SPDLOG_TRACE("Partition not available... Sleeping..");
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
        part.interactions[i] = std::max(part.interactions[i], processed_interactions_[part.idx][i]);
      }
      torch::Tensor tensor = torch::zeros({1}) + 2; // command is 2 for dispatch partition.
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, coordinator_rank_, tag_generator_.getCoordinatorCommandTag());
      if (send_work) {
        send_work->wait();
      }
      part.src = rank_;
      
      SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Dispatched Partition\", 'data':('partition index':{}, 'src':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx, part.src);
      SPDLOG_TRACE("Dispatching partition {}, timestamp: {}. Worker timestamp: {}", part.idx, part.timestamp, timestamp_);

      pb_embeds_->addPartitionForEviction(part.idx);
      pb_embeds_state_->addPartitionForEviction(part.idx);
      
      // TODO(multi_epoch):
      /**
      if (part.T < worker_time_) {
        part.clear_interactions();
        part.increment_timestamp();
      }


      */
      sendPartition(part, coordinator_rank_);
      SPDLOG_TRACE("Dispatched partition {}", part.idx);
    }

    // TODO(multi_epoch): 
    void PrepareForNextEpoch() {
      // // TODO(multi_epoch)
      // signal trainer->setDone() so that isDoneScaling() true;
      // Receive signal from coordinator
      while(timestamp_ < num_epochs_) {
        torch::Tensor tensor = torch::zeros({1});
        std::vector<torch::Tensor> signal({tensor});
        auto recv_work = pg_->recv(signal, coordinator_rank_, tag_generator_.getEpochSignalingTag());
        if (recv_work) {
          recv_work->wait();
          SPDLOG_INFO("Received signal for coordinator for next epoch {}", signal[0].data_ptr<float>()[0]);
          {
            std::lock_guard<std::mutex> guard(next_epoch_mutex_);
            pipeline_->clearCompletedBatches();         
            timestamp_++;
          }
          SPDLOG_INFO("New epoch {} started", timestamp_);
        }
      }
    }

    void flushPartitions(PartitionBuffer* partition_buffer){
      PartitionedFile* partitioned_file = partition_buffer->getPartitionedFile();
      std::vector<Partition *>& partition_table = partition_buffer->getPartitionTable();

      for(int i = 0; i < num_partitions_; i++) {
        if(!partition_table[i]->present_) continue;
        // write partition data from memory to disk but don't clear the partition data pointer
        partitioned_file->writePartition(partition_table[i], false);
      }
    }

    void ServiceEvaluationRequest() {
      while (timestamp_ < num_epochs_) {
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        torch::Tensor source = torch::zeros({num_partitions_}, options);
        std::vector<torch::Tensor> sources({source});
        auto recv_work = pg_->recv(sources, coordinator_rank_, tag_generator_.getEvaluationTag());
        if (recv_work) {
          recv_work->wait();
          SPDLOG_INFO("Received sources from coordinator");

          flushPartitions(pb_embeds_);
          flushPartitions(pb_embeds_state_);

          Timer timer = Timer(gpu_);
          timer.start();
          int recv_partitions_for_eval = 0;
          for (int idx = 0; idx < num_partitions_; idx++) {
            int src = sources[0].data_ptr<int32_t>()[idx];
            SPDLOG_INFO("Partition {} is on worker {}.", idx, src);
            // Receive partition `id` from worker `source`.
            if (src == -1 || src == rank_) {
              // No partition transfer required.
            } else {
              // Transfer partition from worker `source`.
              receivePartition(idx, src);
              recv_partitions_for_eval++;
            }
          }

          timer.stop();
          int64_t event_time = timer.getDuration();
          SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Received Partitions for Evaluations\", 'data':('partitions received':{}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, rank_, recv_partitions_for_eval, rank_, event_time);
    
          // Call evaluator.
          evaluator_->evaluate(true);
        } else {
          throw std::runtime_error("ServiceEvaluationRequest failed to receive evaluation request.");
        }
        // Send evaluation completion message to coordinator.
        torch::Tensor eval = torch::zeros({1});
        std::vector<torch::Tensor> evals({eval});
        auto send_work = pg_->send(evals, coordinator_rank_, tag_generator_.getEvaluationTag());
        if (send_work) {
          send_work->wait();
        } else {
          throw std::runtime_error("ServiceEvaluationRequest failed to send completion message to coordinator.");
        }
      }
    }
    void TransferPartitionsFromWorkerNodes(PartitionMetadata part) {
      if (part.src == -1) {
        // Already have the partition.
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Partition Not Requested\", 'data':('partition index':{}, 'src':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx, part.src);
      } else if (part.src == rank_) {
        // Already have the partition.
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Partition Not Requested\", 'data':('partition index':{}, 'src':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx, part.src);
      } else {
        receivePartition(part.idx, part.src);
      }

      {
        forceToBuffer(pb_embeds_, part.idx);
        forceToBuffer(pb_embeds_state_, part.idx);
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Forced to Buffer\", 'data':('partition index':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx);
      }

      // Add the received partitions to avail_parts_ vector.
      {
        WriteLock w_lock(avail_parts_rw_mutex_);
        SPDLOG_TRACE("Pushed to avail parts: {}", part.idx);
        avail_parts_.push_back(part);
      }
    }
    void receivePartition(int idx, int src) {
      // TODO : add lock to this method
      // Ask the other worker for the partition.
      torch::Tensor request = torch::zeros({1}) + idx;
      std::vector<at::Tensor> request_tensors({request});
      auto send_work = pg_->send(request_tensors, src, tag_generator_.getTagWhenRequesterCommandPath(src));
      if (send_work) {
        send_work->wait();
      }
      
      Timer timer = Timer(gpu_);
      timer.start();
        
      // Receive Partition for node embeddings and optimizer state
      receivePartitionToBuffer(pb_embeds_, src);
      receivePartitionToBuffer(pb_embeds_state_, src);
      
      timer.stop();
      int64_t event_time = timer.getDuration();
      SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Received Partition\", 'data':('partition index':{}, 'src':{}, 'dest': {}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, rank_, idx, src, rank_, event_time);
    }
    void receivePartitionToBuffer(PartitionBuffer *partition_buffer, int src) {
      // Receive metadata
      auto options = torch::TensorOptions().dtype(torch::kInt64);
      torch::Tensor node_embed_tensor = torch::zeros({1, 5}, options);
      std::vector<torch::Tensor> node_embed_tensors({node_embed_tensor});
      auto recv_work = pg_->recv(node_embed_tensors, src, tag_generator_.getTagWhenRequesterDataPath(src));
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
      // 1. Point data_ptr_ to the receive buffer
      partition->data_ptr_ = receive_buffer_;
      
      torch::Tensor tensor_data_recvd = torch::from_blob(receive_buffer_, {partition->partition_size_,partition->embedding_size_},partition->dtype_);              
      // TODO: [Optimization]: class Single large space, any size of partition can be copied there
      // then directly use pwrite to copy to correct portion of node embeddings
      
      // 2. Receive the tensor having data from the worker.
      std::vector<torch::Tensor> tensors_data_recvd({tensor_data_recvd});
      auto recv_data_work = pg_->recv(tensors_data_recvd, src, tag_generator_.getTagWhenRequesterDataPath(src));
      if (recv_data_work) {
        recv_data_work->wait();
      }

      // 3. Write fetched partition to partitioned file.
      std::vector<Partition *>& partition_table = partition_buffer->getPartitionTable();
      PartitionedFile *partition_file = partition_buffer->getPartitionedFile();
      partition_file->writePartition(partition.get());

      partition->data_ptr_ = nullptr;
    }

    void forceToBuffer(PartitionBuffer *partition_buffer, int partition_idx){
      // Admit partition into partition buffer forcefully
      Partition *partition = partition_buffer->getPartitionTable()[partition_idx];
      partition_buffer->admitWithLock(partition);
    }

    void ProcessNewPartition(PartitionMetadata p) {
      SPDLOG_TRACE("Processing new partition: {}, timestamp: {}", p.idx, p.timestamp);
      // Merge local view of partition with global view fetched from co-ordinator
      for(int i = 0; i < num_partitions_; i++){
        processed_interactions_[p.idx][i] = std::max(p.interactions[i], processed_interactions_[p.idx][i]);
        trained_interactions_[p.idx][i] = std::max(p.interactions[i] , trained_interactions_[p.idx][i]);
      }
      // Acquire lock on avail parts and add new batches to be processed to the dataset queue.
      std::vector<pair<int, int>> interactions;
      {
        ReadLock r_lock(avail_parts_rw_mutex_);
        for (const auto &pj : avail_parts_) {
          if (pj.idx == p.idx) {
            if (processed_interactions_[p.idx][pj.idx] <= timestamp_) {
              interactions.push_back({p.idx, pj.idx});
              processed_interactions_[p.idx][pj.idx] = timestamp_+1;
            }
          } else {
            assert(pj.idx != p.idx);
            if (processed_interactions_[p.idx][pj.idx] <= timestamp_ && pj.timestamp == timestamp_) {
              interactions.push_back({p.idx, pj.idx});
              processed_interactions_[p.idx][pj.idx] = timestamp_+1;
            }
            if (processed_interactions_[pj.idx][p.idx] <= timestamp_ && pj.timestamp == timestamp_) {
              interactions.push_back({pj.idx, p.idx});
              processed_interactions_[pj.idx][p.idx] = timestamp_+1;
            }
          }
        }
      }
      if (interactions.size() > 0) { 
        SPDLOG_TRACE("Generated {} interactions", interactions.size());
      }
      for (auto interaction: interactions) {
        int src = interaction.first;
        int dst = interaction.second;
        // Add batch to dataset batches queue. 
        trainset_->addBatchScaling(src, dst);
        SPDLOG_TRACE("Pushed ({}, {}) to dataset queue", src, dst);
      }
    }
    
    void TransferPartitionsToWorkerNodes() {
      // Receive the request and transfer the partition from the node map
      while (timestamp_ < num_epochs_) {
        // Receive request for transfer
        torch::Tensor request = torch::zeros({1});
        std::vector<at::Tensor> tensors({request});
        auto recv_work = pg_->recvAnysource(tensors, tag_generator_.getTagWhenReceiverCommandPath());
        int src_rank;
        if (recv_work) {
          recv_work->wait();
          src_rank = recv_work->sourceRank();
        }
        // send partition metadata
        int part_idx = tensors[0].data_ptr<float>()[0];

        Timer timer = Timer(gpu_);
        timer.start();

        // Send Partition for node embeddings and optimizer state
        sendPartitionFromBuffer(pb_embeds_, src_rank, part_idx);
        sendPartitionFromBuffer(pb_embeds_state_, src_rank, part_idx);

        timer.stop();
        int64_t event_time = timer.getDuration();
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Sent Partition\", 'data':('partition index':{}, 'src': {}, 'dest':{}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part_idx, rank_, src_rank, event_time);
      }
    }

    void sendPartitionFromBuffer(PartitionBuffer* partition_buffer, int src_rank, int partition_index){
        std::vector<Partition*>& partition_table = partition_buffer->getPartitionTable();
        PartitionedFile* partition_file = partition_buffer->getPartitionedFile();
        // metadata: partition_id_, partition_size_, embedding_size_, idx_offset_, file_offset_
        torch::Tensor partition_metadata = partition_table[partition_index]->ConvertMetaDataToTensor();
				std::vector<torch::Tensor> tensors_to_send({partition_metadata});
        // send metadata
        auto send_serialized_partition = pg_->send(tensors_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
        if (send_serialized_partition) {
          send_serialized_partition->wait();
        }                
        // send partition data
        // 1. Read partition from disk
        // 2. Convert to tensor and send
        Partition *partition_to_be_sent = partition_table[partition_index];
        torch::Tensor tensor_data_to_send;
        try {
          if(partition_to_be_sent->present_){
            std::lock_guard<std::mutex> guard(partition_buffer->getAdmitLock());
            tensor_data_to_send = partition_to_be_sent->ConvertDataToTensor();

            std::vector<torch::Tensor> tensors_data_to_send({tensor_data_to_send});
            // send partition data
            auto send_part_data = pg_->send(tensors_data_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
            if (send_part_data){
              send_part_data->wait();
            }
            return;
          }
        } catch(const std::exception& e){
          std::cout << e.what() << std::endl;
        }
        //else
        {
          partition_file->readPartition(send_buffer_, partition_to_be_sent);
          tensor_data_to_send = partition_table[partition_index]->ConvertDataToTensor();
          std::vector<torch::Tensor> tensors_data_to_send({tensor_data_to_send});
          // send partition data
          auto send_part_data = pg_->send(tensors_data_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
          if (send_part_data){
            send_part_data->wait();
          }
        }
        // Can be dispatched to co-ordinator but still be present in the buffer till not actually evicted
        // assert(!partition_table[partition_index]->present_);

    }
    
    void ProcessPartitions() {
      // Generate interactions to be processed.
      // @TODO(scaling): Implement optimal strategies.
      // The strategy computation is expected to be fast and thus we can hold locks
      while (timestamp_ < num_epochs_) {
        std::vector<PartitionMetadata> partitions_done;
        {
          // Lock contention possible as acquired every iteration by this function as well as RequestPartitions
          WriteLock w_lock(avail_parts_rw_mutex_);
          const int avail_parts_size = avail_parts_.size();
          SPDLOG_TRACE("Size of available partitions: {}", avail_parts_size);
          // check if completed --> and set pipeline completedScaling to true

          // Find a suitable candidate for eviction.
          if (avail_parts_size == capacity_) {
            // TODO(multi_epoch): Find partition with timestamp less than worker_time_
            int old_partition_index = -1;
            for (int i = 0; i < avail_parts_size; i++) {
              if (avail_parts_[i].timestamp < timestamp_) {
                old_partition_index = i;
                break;
              }
            }

            std::vector<PartitionMetadata> avail_parts_replacement;
            if (old_partition_index != -1) {
              partitions_done.push_back(avail_parts_[old_partition_index]);
              avail_parts_replacement = avail_parts_;
              avail_parts_replacement.erase(avail_parts_replacement.begin()+old_partition_index);  
            } else {
              for (const auto& part: avail_parts_) assert(part.timestamp == timestamp_);
              // Update trained interaction matrix.
              {
                std::lock_guard<std::mutex> guard(next_epoch_mutex_);
                const int num_batches_processed = pipeline_->getCompletedBatchesSize();
                for (int i = 0; i < num_batches_processed; i++) {
                  PartitionBatch* batch = (PartitionBatch*)pipeline_->completed_batches_[i]; // 
                  int src_idx = batch->src_partition_idx_;
                  int dst_idx = batch->dst_partition_idx_;
                  if (trained_interactions_[src_idx][dst_idx] <= timestamp_ ) {
                    trained_interactions_[src_idx][dst_idx] = timestamp_+1;
                    SPDLOG_TRACE("Trained on partition: ({}, {})", src_idx, dst_idx);
                  }
                }
              }
              // Find partitions with all trained interactions on worker.
              for (int i = 0; i < avail_parts_size; i++) {
                bool done = true;
                const auto& pi = avail_parts_[i];
                for (int j = 0; j < avail_parts_size; j++) {
                  const auto& pj = avail_parts_[j];
                  if (trained_interactions_[pi.idx][pj.idx] <= timestamp_) { done = false; break; }
                  if (trained_interactions_[pj.idx][pi.idx] <= timestamp_) { done = false; break; }
                }
                if (done && partitions_done.empty()) {
                  partitions_done.push_back(pi);
                } else {
                  avail_parts_replacement.push_back(pi);
                }
              }
            }

            // For debugging
            if(avail_parts_.size() != avail_parts_replacement.size()){
              SPDLOG_TRACE("Available parts changed...");
              SPDLOG_TRACE("Old available parts:");
              for(int i = 0; i < avail_parts_.size(); i++) SPDLOG_TRACE("{}", avail_parts_[i].idx);

              SPDLOG_TRACE("New available parts:");
              for (int i = 0; i < avail_parts_replacement.size(); i++) SPDLOG_TRACE("{}", avail_parts_replacement[i].idx);
            }
            // Update avail_parts.
            avail_parts_ = avail_parts_replacement;
          }
        }

        // TODO: Introduce a queue and put this into separate thread
        for (auto p : partitions_done) {
          // queue.push(p)
          SPDLOG_TRACE("Dispatching partition {} to co-ordinator..", p.idx);
          DispatchPartitionsToCoordinator(p);
        }
        // sleep
        // Increase sleep time to reduce contention for lock
        // TODO(rrt): Replace this by a condition variable.
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
      }
    }
    
    void RunTrainer() {
      trainer_->train(num_epochs_);
      // TODO(multi_epoch)
      // trainer_->trainScaling()
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
  mutable std::mutex next_epoch_mutex_;
  int partition_size_;
  int dtype_size_;
  int embedding_size_; 
  int embedding_dims_;
  int timestamp_;
  int num_epochs_;
  bool gpu_;
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
  std::string perf_metrics_label_;
};


int main(int argc, char* argv[]) {
  marius_options = parseConfig(argc, argv); // marius options is an extern variable form config.h that is globally used across the library.
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
  string base_dir = "/mnt/data/Work/marius";
  auto filestore = c10::make_intrusive<c10d::FileStore>(base_dir + "/rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);
  // auto dev = c10d::GlooDeviceFactory::makeDeviceForInterface("lo");
  std::chrono::hours timeout(24);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("lo"));
  options->timeout = timeout;
  options->threads = options->devices.size() * 2;
  auto pg = std::make_shared<c10d::ProcessGroupGloo>(
    prefixstore, rank, world_size, options);
  int num_partitions = marius_options.storage.num_partitions;
  int capacity = marius_options.storage.buffer_capacity;
  bool gpu = false;
  // if (marius_options.general.device == torch::kCUDA) {
  //     gpu = true;
  // }
  WorkerNode worker(pg, rank, capacity, num_partitions, world_size-1, marius_options.training.num_epochs, gpu);
  std::string log_file = marius_options.general.experiment_name;
  MariusLogger marius_logger = MariusLogger(log_file);
  spdlog::set_default_logger(marius_logger.main_logger_);
  marius_logger.setConsoleLogLevel(marius_options.reporting.log_level);
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
9. Convert Asserts to runtime errors
*/