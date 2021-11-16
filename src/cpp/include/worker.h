#ifndef MARIUS_WORKER_H
#define MARIUS_WORKER_H


#include "message.h"
#include "config.h"
#include "storage.h"

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

class CommWorker {
public:
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
    std::vector<std::vector<int>> processed_interactions_;
    int partition_size_;
    int embedding_dims_;

    // PartitionBufferStorage* embeds_;
    // PartitionBufferStorage* embeds_state_;
    // PartitionBuffer* pb_embeds_;
    // PartitionBuffer* pb_embeds_state_;
    CommWorker(MariusOptions marius_options);
    CommWorker(std::shared_ptr<c10d::ProcessGroupGloo> pg, int rank, int capacity, int num_partitions, int num_workers);
    PartitionMetadata RequestPartition();
    // Partition* getPartitionFromWorkerOrDisk(PartitionMetadata part);
};

#endif