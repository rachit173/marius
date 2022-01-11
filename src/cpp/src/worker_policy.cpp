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


// Worker policy 1: Decide order of interactions to be processed
void WorkerNode::orderInteractions(std::vector<pair<int, int>>& interactions) {
    // Default policy: lexicographic order of pair
}

// Worker policy 2: Decide which partition should be evicted
void WorkerNode::ProcessPartitions() {
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
      SPDLOG_TRACE("Dispatching partition {} to co-ordinator..", p.idx);
      DispatchPartitionsToCoordinator(p);
    }
    // sleep
    // Increase sleep time to reduce contention for lock
    // TODO(rrt): Replace this by a condition variable.
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
  }
}

