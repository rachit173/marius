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

#include "worker.h"

typedef std::shared_mutex RwLock;
typedef std::unique_lock<RwLock> WriteLock;
typedef std::shared_lock<RwLock> ReadLock;
namespace fs = std::filesystem;

CommWorker::CommWorker(std::shared_ptr<c10d::ProcessGroupGloo> pg, int rank, int capacity, int num_partitions, int num_workers){
    this->pg_= pg;
    this->rank_= rank;
    this->capacity_= capacity_;
    this->num_partitions_= num_partitions;
    this->num_workers_= num_workers;
    coordinator_rank_ = num_workers_; 
    sleep_time_ = 100; // us
    processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
}

PartitionMetadata CommWorker::RequestPartition()
{
    torch::Tensor tensor = torch::zeros({1}) + 1; // command is 1 for request partition.
    std::vector<at::Tensor> tensors({tensor});
    std::cout << "Sending request for getting partition metadata from co-ordinator..." << std::endl;
    auto send_work = pg_->send(tensors, coordinator_rank_, 0);
    if (send_work) {
        send_work->wait();
    }
    std::cout << "Started Receiving" << std::endl;
    // Receive the returned partition
    torch::Tensor part_tensor = torch::zeros({2});
    std::vector<at::Tensor> part_tensor_vec({part_tensor});
    auto recv_work = pg_->recv(part_tensor_vec, coordinator_rank_, 0);
    if (recv_work)
    {
        recv_work->wait();
    }
    std::cout << "Received " << part_tensor_vec[0] << std::endl;
    return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
}

CommWorker::CommWorker(MariusOptions marius_options){
    int num_partitions = marius_options.storage.num_partitions;
    int capacity = marius_options.storage.buffer_capacity;
    
    int rank = marius_options.communication.rank;
    int world_size = marius_options.communication.world_size;
    std::string prefix = marius_options.communication.prefix;

    std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
    
    auto filestore = c10::make_intrusive<c10d::FileStore>("/proj/uwmadison744-f21-PG0/file", 1);
    auto prefixstore = c10::make_intrusive<c10d::PrefixStore>("abc", filestore);
    std::chrono::milliseconds timeout(100000);
    auto options = c10d::ProcessGroupGloo::Options::create();
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForInterface("eth1"));
    options->timeout = timeout;
    options->threads = options->devices.size() * 2;
    auto pg = std::make_shared<c10d::ProcessGroupGloo>(prefixstore, rank, world_size, options);

    pg_ = pg;
    rank_ = rank;
    capacity_ = capacity;
    num_partitions_ = num_partitions;
    num_workers_ = world_size - 1;
    coordinator_rank_ = num_workers_; 
    sleep_time_ = 100; // us
    processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
}

// Partition* CommWorker::getPartitionFromWorkerOrDisk(PartitionMetadata part) {
//     if (part.src == -1) {
//         std::cout << "Need to fetch it from current storage..." << std::endl;
//         return nullptr;
//     }
//     else
//     {
//         int tag = 1;
//         // ask the other worker for the partition
//         torch::Tensor request = torch::zeros({1}) + part.idx;
//         std::vector<at::Tensor> request_tensors({request});
//         auto send_work = pg_->send(request_tensors, part.src, tag);
//         if (send_work)
//         {
//             send_work->wait();
//         }
//         // Receive the tensor from the worker.
//         std::vector<at::Tensor> node_embed_tensors({torch::zeros({partition_size_, embedding_dims_})});
//         auto recv_work = pg_->recv(node_embed_tensors, part.src, tag);
//         if (recv_work)
//         {
//             recv_work->wait();
//         }
//         // Set the received tensor to the node map
//         node_map[part.idx] = std::make_shared<torch::Tensor>(std::move(node_embed_tensors[0]));
//     }
//     // Add the received partitions to avail_parts_ vector.
//     {
//         WriteLock w_lock(avail_parts_rw_mutex_);
//         avail_parts_.push_back(part);
//     }

//     return nullptr;
// }