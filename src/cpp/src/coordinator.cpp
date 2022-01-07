#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"
#include "communication.h"
#include "coordinator.h"

#include <iostream>
#include <vector>
#include <memory>
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

//######### Handlers for partition allocate and receive ########################
void Coordinator::handlePartitionAllocate(int srcRank) {
  PartitionMetadata part = AllocatePartition(srcRank);
  sendPartition(part, srcRank);
  if (part.idx != -1) {
    in_process_partitions_[srcRank][part.idx] = 1;
  }
}

void Coordinator::handlePartitionReceive(int srcRank) {
  PartitionMetadata part = receivePartition(srcRank);
  if (part.timestamp < timestamp_) {
    part.updateTimestamp(timestamp_);
  }
  assert(part.src == srcRank);
  syncPartitionMetadata(part);
}
// ################################################################
//###################### Handle Epoch completion #####################################
double Coordinator::getCompletionRatio(int ts) {
  double total = num_partitions_ * num_partitions_;
  double completed = 0; 
  for (int i = 0; i < num_partitions_; i++) {
    for (int j = 0; j < num_partitions_; j++) {
      if (processed_interactions_[i][j] == ts+1) completed+=1;
    }
  }
  return completed / total;
}

void Coordinator::updateInteractionsTimestamp(int ts) {
  for (int i = 0; i < num_partitions_; i++) {
    for (int j = 0; j < num_partitions_; j++) {
      processed_interactions_[i][j]=ts;
    }
  }      
}

void Coordinator::updateAvailablePartitionsTimestamp(int ts) {
  for (auto& part: available_partitions_) {
    part.updateTimestamp(ts);
  }
}

void Coordinator::Evaluation() {
  SPDLOG_INFO("Performing evaluation");
  // 1. Identify the source workers for each node partition `sources`.
  int sources[num_partitions_];
  // in_process_partitions_ and avaible_partitions_ 
  // are mutually exclusive and exhaustive for the node partitions.
  for (int rank = 0; rank < num_workers_; rank++) {
    for (int id = 0; id < num_partitions_; id++) {
      if (in_process_partitions_[rank][id]==1) {
        sources[id] = rank;
      }
    }
  }
  for (const auto& part : available_partitions_) {
    sources[part.idx] = part.src;
  }
  // 2. Send the `sources` array to worker 0.
  auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto tensor = torch::from_blob(sources, {num_partitions_}, options).clone();
  std::vector<torch::Tensor> tensors({tensor});
  int dstRank = 0;
  auto send_work = pg_->send(tensors, dstRank, tag_generator_.getWorkerSpecificEvaluationTag(dstRank));
  if (send_work) {
    send_work->wait();
  } else {
    throw std::runtime_error("Evaluation request to worker 0 failed");
  }
  // 3. Wait for worker 0 response containing evaluation.
  torch::Tensor eval_tensor = torch::zeros({1});
  std::vector<torch::Tensor> eval_tensors({eval_tensor});
  int srcRank = 0;
  auto recv_work = pg_->recv(eval_tensors, srcRank, tag_generator_.getWorkerSpecificEvaluationTag(srcRank));
  if (recv_work) {
    recv_work->wait();
  } else {
    throw std::runtime_error("Evaluation failed, did not receive response from worker 0.");
  }
}

void Coordinator::HandleEpochCompletion() {
    if (getCompletionRatio(timestamp_) < 0.999) return;

    epoch_timer_.stop();
    int64_t epoch_time = epoch_timer_.getDuration();

    SPDLOG_DEBUG("{} ('event':\"Epoch Runtime\", 'data':('epoch':{}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, (double) epoch_time / 1000);
    // TODO(scaling): Replace 3 by epochs_per_eval.
    if (timestamp_% epochs_per_eval_ == 0 || timestamp_+1 == num_epochs_) {
        // Blocks till worker 0 does not return the evaluation.
        Evaluation();
    }
    timestamp_++;
    updateInteractionsTimestamp(timestamp_);
    updateAvailablePartitionsTimestamp(timestamp_);
    epoch_timer_.start();
    // Signal all workers for next epoch
    for (int rank = 0; rank < num_workers_; rank++) {
        bool signal_success = signalNextEpoch(rank);
        if(signal_success){
          SPDLOG_TRACE("Signalled {}", rank);
        }
    }
}

bool Coordinator::signalNextEpoch(int dstRank) {
  auto tensor = torch::zeros({1}) + timestamp_;
  std::vector<torch::Tensor> tensors({tensor});
  auto send_work = pg_->send(tensors, dstRank, tag_generator_.getWorkerSpecificEpochSignalingTag(dstRank));
  if (send_work) {
    send_work->wait();
  } else {
    return false;
  }
  return true;
}
//###################################################################################

//######################### Communication code #######################################
bool Coordinator::sendPartition(PartitionMetadata part, int dstRank) {
    const auto& tensor = part.ConvertToTensor();
    std::vector<at::Tensor> tensors({tensor});
    auto send_work = pg_->send(tensors, dstRank, tag_generator_.getWorkerSpecificCommunicationTag(dstRank));
    if (send_work) {
        send_work->wait();
    } else {
        return false;
    }
    return true;
}

PartitionMetadata Coordinator::receivePartition(int srcRank) {
    torch::Tensor part_tensor = torch::zeros({num_partitions_+kPartititionMetadataSerde});
    std::vector<torch::Tensor> part_tensor_vec({part_tensor});
    auto recv_work = pg_->recv(part_tensor_vec, srcRank, tag_generator_.getWorkerSpecificCommunicationTag(srcRank));
    if (recv_work) {
      recv_work->wait();
    }
    // SPDLOG_TRACE ("Tensor received : {}", part_tensor_vec[0]);
    return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
}
//###################################################################################

//////////////////////////////////////////////////////////////////////////////////////////
void Coordinator::printCoordinatorState() {
  std::cout << "Available partitions at coordinator - " << std::endl;
  for(const auto& itr: available_partitions_){
    std::cout << "Partition Index: " << itr.idx << " -->  Owner: " << itr.src << std::endl;
  }
  std::cout << "Processed Interactions: " << std::endl;
  for(int i = 0; i < num_partitions_; i++){
    for(int j = 0; j < num_partitions_; j++){
      std::cout << processed_interactions_[i][j] << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "In-process partitions: " << std::endl;
  for(int i = 0; i < num_workers_; i++){
    for(int j = 0; j < num_partitions_; j++){
      std::cout << in_process_partitions_[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}

void Coordinator::start_working() {
  int command;
  int srcRank = -1;
  epoch_timer_.start();

  while (timestamp_ < num_epochs_) {
    torch::Tensor tensor = torch::zeros({1});
    std::vector<at::Tensor> tensors({tensor});
    auto recv_work = pg_->recvAnysource(tensors, tag_generator_.getCoordinatorCommandTag());
    if (recv_work) {
      recv_work->wait();
      srcRank = recv_work->sourceRank();
    }

    command = tensors[0].data_ptr<float>()[0];
    SPDLOG_TRACE("Command {} From {}", command, srcRank);

    if (command == ALLOCATE_PARTITION) {
      handlePartitionAllocate(srcRank);
    } else if (command == RECEIVE_PARTITION) {
      handlePartitionReceive(srcRank);
    } else {
      SPDLOG_WARN("Received an invalid command: {}", command);
    }
    printCoordinatorState();
    HandleEpochCompletion();
  }
}

int coordinator_main(int argc, char* argv[]) {
  auto marius_options = parseConfig(argc, argv);
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  int epochs_per_eval = marius_options.evaluation.epochs_per_eval;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
  std::cout << "Total epochs: " << marius_options.training.num_epochs << std::endl;

  std::string iface = marius_options.communication.iface;
  std::string MASTER_IP = marius_options.communication.master;
  int MASTER_PORT = 29501;
  auto tcpstore = c10::make_intrusive<c10d::TCPStore>(MASTER_IP, MASTER_PORT, 1, true);
  // auto filestore = c10::make_intrusive<c10d::FileStore>(base_dir + "/rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", tcpstore);

  // Setup logging for coordinator
  std::string log_file = marius_options.general.experiment_name + "_coordinator";
  MariusLogger marius_logger = MariusLogger(log_file);
  spdlog::set_default_logger(marius_logger.main_logger_);
  marius_logger.setConsoleLogLevel(marius_options.reporting.log_level);
  
  // Required for timer
  bool gpu = false;
  // if (marius_options.general.device == torch::kCUDA) {
  //     gpu = true;
  // }
  std::chrono::hours timeout(24);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface(iface));
  options->timeout = timeout;
  options->threads = options->devices.size() * 2;
  auto pg = std::make_shared<c10d::ProcessGroupGloo>(
    prefixstore, rank, world_size, options);
  int num_partitions = marius_options.storage.num_partitions;
  Coordinator coordinator(pg, num_partitions, world_size-1, marius_options.training.num_epochs, epochs_per_eval, gpu);
  coordinator.start_working();
  coordinator.stop_working();
  return 0;
}