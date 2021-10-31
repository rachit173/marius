#include "message.h"

#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include <thread>
#include <shared_mutex>
#include <mutex>

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

class Worker {
  public:
    explicit Worker(
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
      // setup
    }
    void start_working() {
      // Need 
      // 1. Request partitions when below capacity.
      // 2. Dispath partitions
      // 3. Transfer partitions to other workers
      threads_.emplace_back(std::thread([&](){
        this->RequestPartitions();
      }));
      threads_.emplace_back(std::thread([&](){
        this->DispatchPartitionsToCoordinator();
      }));
      threads_.emplace_back(std::thread([&](){
        this->TransferPartitionsToWorkers();
      }));
      threads_.emplace_back(std::thread([&](){
        this->ProcessPartitions();
      }));
      for (auto& t: threads_) {
        t.join();
      }
    }
    void stop_working() {}
  private:
    Partition RequestParition() {
      return Partition(0);
    }
    void RequestPartitions() {
      while (1) {
        // if ()
        int size;
        {
          ReadLock r_lock(avail_parts_rw_mutex_);
          size = avail_parts_.size();
        }
        while (size < capacity_) {
          Partition p = RequestParition();
          {
            WriteLock w_lock(transfer_receive_parts_rw_mutex_);
            transfer_receive_parts_.push(p);
          }
          size++;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
      }
    }
    void DispatchPartitionsToCoordinator() {
      while (1) {
        {
          ReadLock r_lock(dispatch_parts_rw_mutex_);

        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
      }
    }
    void TransferPartitionsFromWorkers() {

    }
    void TransferPartitionsToWorkers() {

    }
    void ProcessPartitions() {

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
  std::vector<Partition> avail_parts_;
  mutable std::shared_mutex dispatch_parts_rw_mutex_;
  std::queue<Partition> dispatch_parts_;
  mutable std::shared_mutex transfer_receive_parts_rw_mutex_;
  std::queue<Partition> transfer_receive_parts_;
  std::vector<std::thread> threads_;
};


int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "The command should be run as ./a.aout <prefix> <rank> <size>" << std::endl;
    return 0;
  }
  std::string prefix(argv[1]);
  const int rank = atoi(argv[2]);
  const int world_size = atoi(argv[3]);
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
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
  int num_partitions = 8;
  int capacity = 4;
  Worker worker(pg, rank, capacity, num_partitions, world_size-1);
  worker.start_working();
  worker.stop_working();
  // if (rank == 1) {
  //   std::cout << "Receiving" << std::endl;
  //   torch::Tensor tensor = torch::zeros({2, 3});
  //   std::vector<at::Tensor> tensors({tensor});
  //   int srcRank = -1;
  //   int tag = 0;
  //   auto recv_work = pg->recvAnysource(tensors, tag);
  //   if (recv_work) {
  //     recv_work->wait();
  //     srcRank = recv_work->sourceRank();
  //   }
  //   std::cout << "Received from " << srcRank << std::endl;
  //   for (auto tensor : tensors) {
  //     std::cout << tensor << ", " << std::endl;
  //   }
  //   std::cout << std::endl;
  // } else {
  //   std::cout << "Sending" << std::endl;
  //   torch::Tensor tensor = torch::rand({2, 3});
  //   std::vector<at::Tensor> tensors({tensor});
  //   int dstRank = 1;
  //   int tag = 0;
  //   auto work = pg->send(tensors, dstRank, tag);
  //   if (work) work->wait();
  // }
}