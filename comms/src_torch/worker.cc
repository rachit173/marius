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



///



//
// TODO List:
// 1. Process group (pg_) is not thread safe, requires locking.
// 2. Multiple process groups are needed, atleast one for 
// metadata communication and one for bulk transfer.
// 3. Interactions need to be transferred to coordinator.
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
      int partition_size_ = 10;
      int embedding_dims_ = 12;
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
    Partition RequestPartition() {
      torch::Tensor tensor = torch::zeros({1}) + 1; // command is 1 for request partition.
      std::vector<at::Tensor> tensors({tensor});
      auto send_work = pg_->send(tensors, coordinator_rank_, 0);
      if (send_work) {
        send_work->wait();
      }
      std::cout << "Started Receiving" << std::endl;
      // Receive the returned partition
      torch::Tensor part_tensor = torch::zeros({2});
      std::vector<at::Tensor> part_tensor_vec({part_tensor});
      auto recv_work = pg_->recv(part_tensor_vec, coordinator_rank_, 0);
      if (recv_work) {
        recv_work->wait();
      }
      std::cout << "Received " << part_tensor_vec[0] << std::endl;
      return Partition::ConvertToPartition(part_tensor_vec[0]);
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
          Partition p = RequestPartition();
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
    bool sendPartition(const Partition& part, int dstRank) {
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
          WriteLock w_lock(dispatch_parts_rw_mutex_);
          if (dispatch_parts_.empty()) break;
          Partition part = dispatch_parts_.front();
          dispatch_parts_.pop();
          sendPartition(part, coordinator_rank_);      
        }
        // TODO: all sleep based code can be converted to conditional
        // variable.
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
      }
    }
    void fetchFromStorage(int part_num) {
      // For now just generate a random embedding vector.
      node_map[part_num] = std::make_shared<torch::Tensor>(torch::randn({partition_size_, embedding_dims_}));
    }
    void TransferPartitionsFromWorkers() {
      while (1) {
        {
          while (1) {
            WriteLock w_lock(transfer_receive_parts_rw_mutex_);
            if (transfer_receive_parts_.empty()) break;
            Partition part  = transfer_receive_parts_.front();
            if (part.src == -1) {
              // Needs to fetch it from current storage 
              fetchFromStorage(part.idx);
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
              // set the received tensor to the node map
              node_map[part.idx] = std::make_shared<torch::Tensor>(std::move(node_embed_tensors[0]));
            }
          }
          // TODO: all sleep based code can be converted to conditional
          // variable.
          std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_)); // TODO(rrt): reduce this later;
        }
        
      }
    }
    void TransferPartitionsToWorkers() {
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
        std::vector<at::Tensor> tensors;
        tensors.push_back(std::move(*node_map[part_idx]));
        pg_->send();



        // empty that entry in the node map after sending
        node_map[part_idx] = nullptr;
      }
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
  mutable std::shared_mutex node_map_rw_mutex_;
  std::vector<std::shared_ptr<torch::Tensor>> node_map;
  int partition_size_;
  int embedding_dims_;
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