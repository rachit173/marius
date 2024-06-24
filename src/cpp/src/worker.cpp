#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"
#include "communication.h"

#include "worker.h"

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
#include <c10d/TCPStore.hpp>
#include <c10d/PrefixStore.hpp>
#include <c10d/Store.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/GlooDeviceFactory.hpp>
#include <c10d/frontend.hpp>

namespace fs = std::filesystem;

// TODO List:
// 1. Process group (pg_) is not thread safe, requires locking.
// 2. Multiple process groups are needed, atleast one for 
// metadata communication and one for bulk transfer.
  
void WorkerNode::start_working(DataSet* trainset, DataSet* evalset, 
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
  }

  // Run actual training pipeline
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

  // Request partitions when below capacity.
  threads_.emplace_back(std::thread([&](){
    this->RequestPartitions();
  }));

  // Mark partitions when training done and dispatch partition metadata to co-ordinator
  threads_.emplace_back(std::thread([&](){
    this->ProcessPartitions();
  }));

  // Transfer partitions to other workers
  threads_.emplace_back(std::thread([&](){
    this->TransferPartitionsToWorkerNodes();
  }));

  // Transition into new epoch
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

/* Request partitions
1. Get available partition from co-ordinator
2. Get partition data from another worker or disk
3. Feed data to the training pipeline using the partition
*/
void WorkerNode::RequestPartitions() {
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

void WorkerNode::PrepareForNextEpoch() {
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

void WorkerNode::ServiceEvaluationRequest() {
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

void WorkerNode::RunTrainer() {
  trainer_->train(num_epochs_);
}

int worker_main(int argc, char* argv[]) {
  marius_options = parseConfig(argc, argv); // marius options is an extern variable form config.h that is globally used across the library.
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
  
  std::string iface = marius_options.communication.iface;
  std::string MASTER_IP = marius_options.communication.master;
  int MASTER_PORT = 29501;
  auto tcpstore = c10::make_intrusive<c10d::TCPStore>(MASTER_IP, MASTER_PORT, 1, false);
  // auto filestore = c10::make_intrusive<c10d::FileStore>(base_dir + "/rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", tcpstore);
  // auto dev = c10d::GlooDeviceFactory::makeDeviceForInterface("lo");
 
  std::chrono::hours timeout(24);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface(iface));
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
  std::string log_file = marius_options.general.experiment_name + "_worker_" + std::to_string(rank);
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
  return 0;
}

/*TODO:
1. Reduce interactions by a better policy
2. Run in twitter data to observe bottlenecks
3. Prefetching fix
4. Convert Asserts to runtime errors
*/