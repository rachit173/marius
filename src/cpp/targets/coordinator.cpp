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
      for (int i = 0; i < num_partitions_; i++) {
        available_partitions_.push(PartitionMetadata(i, -1, num_partitions_));
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
        std::cout << "Received " << tensors[0] << " from " << srcRank << std::endl;
        command = tensors[0].data_ptr<float>()[0];
        std::cout << "Command: " << command << std::endl;
        if (command == 1) {
          PartitionMetadata part = PartitionRequest(srcRank);
          if (part.idx != -1) sendPartition(part, srcRank);
        } else if (command == 2) {
          PartitionMetadata part = receivePartition(srcRank);
          available_partitions_.push(part);
        } else {
          std::cout << "Received an invalid command: " << command << "\n";       
        }
      }
    }
    void stop_working() {

    }
  private:
    // TODO(scaling): Move to PartitionMetadata
    bool sendPartition(PartitionMetadata part, int dstRank) {
      const auto& tensor = part.ConvertToTensor();
      std::cout << "tensor to send - " << tensor << std::endl;
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
      if (available_partitions_.empty()) {
        std::cout << "No part available" << std::endl;
        return PartitionMetadata(-1, -1, num_partitions_);
      }
      PartitionMetadata part = available_partitions_.front();
      available_partitions_.pop();
      return part;
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
  std::queue<PartitionMetadata> available_partitions_;
  CoordinatorTagGenerator tag_generator_;
};


int main(int argc, char* argv[]) {
  auto marius_options = parseConfig(argc, argv);
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;

  auto filestore = c10::make_intrusive<c10d::FileStore>("./rendezvous_checkpoint", 1);
  auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);

  std::chrono::milliseconds timeout(100000);
  auto options = c10d::ProcessGroupGloo::Options::create();
  options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("lo"));
  options->timeout = timeout;
  options->threads = options->devices.size() * 2;
  auto pg = std::make_shared<c10d::ProcessGroupGloo>(
    prefixstore, rank, world_size, options);
  int num_partitions = marius_options.storage.num_partitions;
  Coordinator coordinator(pg, num_partitions, world_size-1);
  coordinator.start_working();
  coordinator.stop_working();
}