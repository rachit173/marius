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
#include <exception>
#include <filesystem>

#include <torch/torch.h>
#include <c10d/FileStore.hpp>
#include <c10d/PrefixStore.hpp>
#include <c10d/Store.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/GlooDeviceFactory.hpp>
#include <c10d/frontend.hpp>
namespace fs = std::filesystem;

class Coordinator {
  public:
    explicit Coordinator(
      std::shared_ptr<c10d::ProcessGroupGloo> pg,
      int num_partitions,
      int num_workers
    ):
    pg_(pg), 
    num_partitions_(num_partitions),
    num_workers_(num_workers),
    tag_generator_(num_workers) {
      // setup
      available_partitions_.clear();
      in_process_partitions_.resize(num_workers_, vector<int>(num_partitions_, 0));
      processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      for (int i = 0; i < num_partitions_; i++) {
        available_partitions_.push_back(PartitionMetadata(i, -1, num_partitions_));
      }
    }
    void start_working() {
      while (1) {
        std::cout << "Receiving" << std::endl;
        torch::Tensor tensor = torch::zeros({1});
        std::vector<at::Tensor> tensors({tensor});
        int command = tensors[0].data_ptr<float>()[0];

        int srcRank = -1;
        auto recv_work = pg_->recvAnysource(tensors, tag_generator_.getCoordinatorCommandTag());
        if (recv_work) {
          recv_work->wait();
          srcRank = recv_work->sourceRank();
        }
        // std::cout << "Received " << tensors[0] << " from " << srcRank << std::endl;
        command = tensors[0].data_ptr<float>()[0];
        std::cout << "Command: " << command << " From: " << srcRank << std::endl;

        if (command == 1) {
          PartitionMetadata part = PartitionRequest(srcRank);
          sendPartition(part, srcRank);
          if (part.idx != -1) {
            in_process_partitions_[srcRank][part.idx] = 1;
          }
        } else if (command == 2) {
          PartitionMetadata part = receivePartition(srcRank);
          assert(part.src == srcRank);
          // update co-ordinator view of partitions and interactions
          available_partitions_.push_back(part);
          in_process_partitions_[part.src][part.idx] = 0;
          syncInteractions(part);
        } else {
          std::cout << "Received an invalid command: " << command << "\n";       
        }
        printCoordinatorState();
      }
    }
    void stop_working() {

    }
  private:
    void syncInteractions(const PartitionMetadata& part){
      // TODO(rrt): Update the symmetric actions that have happened at that worker as well.
      for(int i = 0; i < num_partitions_; i++){
        processed_interactions_[part.idx][i] = processed_interactions_[part.idx][i] | part.interactions[i];
        processed_interactions_[i][part.idx] = processed_interactions_[i][part.idx] | part.interactions[i];
      }
    }

    void printCoordinatorState() {
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
    
    // TODO(scaling): Move to PartitionMetadata
    bool sendPartition(PartitionMetadata part, int dstRank) {
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
    
    PartitionMetadata PartitionRequest(int srcRank) {
      // Assumes that total # of partitions >=  # of workers * Buffer capacity per worker
      assert(!available_partitions_.empty());

      // Dispatch partition which can do maximum interactions considering the existing state of worker and already done interactions
      // and take the maximum
      vector<int> possible_interactions(num_partitions_, 0);
      int max_interactions = 0;
      PartitionMetadata response_partition(-1, -1, num_partitions_);
      for(const auto& p: available_partitions_){
        for(int j = 0; j < num_partitions_; j++){
          if(j == p.idx){
            possible_interactions[p.idx] += (processed_interactions_[p.idx][j] == 0);
          } else if(in_process_partitions_[srcRank][j] == 1){
            // if not processed, add as an interaction. Can add symmetric interaction to double, but no use....
            possible_interactions[p.idx] += 2 * (processed_interactions_[p.idx][j] == 0);
          }
        }
        if (possible_interactions[p.idx] > max_interactions) {
          max_interactions = possible_interactions[p.idx];
          response_partition = p;
        }
      }

      // Delete if partition present
      if(response_partition.idx != -1){
        for(auto itr = available_partitions_.begin(); itr != available_partitions_.end(); itr++){
          if(itr->idx == response_partition.idx){
            available_partitions_.erase(itr);
            break;
          }
        }
      } else {
        
        // Just send a "random" partition
        int idx = rand() % available_partitions_.size();
        response_partition = available_partitions_[idx];
        available_partitions_.erase(available_partitions_.begin() + idx);
      }
      return response_partition;
      // TODO: If nothing was available from other workers' partitions --> Send partition owned by the same worker

      // Not available
      SPDLOG_INFO("Partitions not available for worker with rank: {}", srcRank);
      return PartitionMetadata(-1, -1, num_partitions_);
    }
    // TODO(scaling): Move to PartitionMetadata
    PartitionMetadata receivePartition(int srcRank) {
      torch::Tensor part_tensor = torch::zeros({num_partitions_+3});
      std::vector<at::Tensor> part_tensor_vec({part_tensor});
      auto recv_work = pg_->recv(part_tensor_vec, srcRank, tag_generator_.getWorkerSpecificCommunicationTag(srcRank));
      if (recv_work) {
        recv_work->wait();
      }
      std::cout << "tensor received " << part_tensor_vec[0] << std::endl;
      return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
    }


  private:
  std::shared_ptr<c10d::ProcessGroupGloo> pg_;
  int num_partitions_;
  int num_workers_;
  std::vector<PartitionMetadata> available_partitions_;
  std::vector<vector<int>> in_process_partitions_;
  std::vector<vector<int>> processed_interactions_;
  CoordinatorTagGenerator tag_generator_;
};


int main(int argc, char* argv[]) {
  auto marius_options = parseConfig(argc, argv);
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;

  string base_dir = "/proj/uwmadison744-f21-PG0/groups/g007";
  auto filestore = c10::make_intrusive<c10d::FileStore>(base_dir + "/rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);

  std::chrono::milliseconds timeout(10000000);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("eth1"));
  options->timeout = timeout;
  options->threads = options->devices.size() * 2;
  auto pg = std::make_shared<c10d::ProcessGroupGloo>(
    prefixstore, rank, world_size, options);
  int num_partitions = marius_options.storage.num_partitions;
  Coordinator coordinator(pg, num_partitions, world_size-1);
  coordinator.start_working();
  coordinator.stop_working();
}