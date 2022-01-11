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
#include <c10d/ProcessGroupGloo.hpp>


void WorkerNode::ClearProcessedInteractions() {
    assert(processed_interactions_.size() == num_partitions_);
    assert(processed_interactions_[0].size() == num_partitions_);
    for (int i = 0; i < num_partitions_; i++) {
        for (int j = 0; j < num_partitions_; j++) {
            processed_interactions_[i][j] = 0;
        }
    }
}

int WorkerNode::getSize(){
    ReadLock r_lock(avail_parts_rw_mutex_);
    int size = avail_parts_.size();
    r_lock.unlock();
    return size;
}

void WorkerNode::ProcessNewPartition(PartitionMetadata p) {
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

    orderInteractions(interactions);

    for (auto interaction: interactions) {
        int src = interaction.first;
        int dst = interaction.second;
        // Add batch to dataset batches queue. 
        trainset_->addBatchScaling(src, dst);
        SPDLOG_TRACE("Pushed ({}, {}) to dataset queue", src, dst);
    }
}

// #################################################################################3


void WorkerNode::DispatchPartitionsToCoordinator(PartitionMetadata part) {
    // PartitionMetadata -> interactions ==> processed_interactions_row for index part.idx
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

    // Put into eviction queue which belongs to partition buffer
    pb_embeds_->addPartitionForEviction(part.idx);
    pb_embeds_state_->addPartitionForEviction(part.idx);
    
    sendPartition(part, coordinator_rank_);
    SPDLOG_TRACE("Dispatched partition {}", part.idx);
}

void WorkerNode::updateProcessedPartitions() {
    // Update trained interaction matrix.
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

void WorkerNode::printPartitionChange(vector<int>& avail_parts_replacement) {
    // For debugging
    if(avail_parts_.size() != avail_parts_replacement.size()){
        SPDLOG_TRACE("Available parts changed...");
        SPDLOG_TRACE("Old available parts:");
        for(int i = 0; i < avail_parts_.size(); i++) SPDLOG_TRACE("{}", avail_parts_[i].idx);

        SPDLOG_TRACE("New available parts:");
        for (int i = 0; i < avail_parts_replacement.size(); i++) SPDLOG_TRACE("{}", avail_parts_replacement[i].idx);
    }
}
// #####################################################################################

void WorkerNode::flushPartitions(PartitionBuffer* partition_buffer) {
    PartitionedFile* partitioned_file = partition_buffer->getPartitionedFile();
    std::vector<Partition *>& partition_table = partition_buffer->getPartitionTable();

    for(int i = 0; i < num_partitions_; i++) {
        if(!partition_table[i]->present_) continue;
        // write partition data from memory to disk but don't clear the partition data pointer
        partitioned_file->writePartition(partition_table[i], false);
    }
}
