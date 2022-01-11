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

// TODO: Decouple actual policy implementation from interface
// Worker policy 2: Decide which partition should be evicted
void WorkerNode::evictPartitions(vector<int>& to_be_evicted, vector<int>& to_be_retained) {
  
  
  int avail_parts_size = to_be_retained.size();
  // Find partitions with all trained interactions on worker.
  for (int i = 0; i < avail_parts_size; i++) {
    bool done = true;
    const auto& pi = avail_parts_[i];
    for (int j = 0; j < avail_parts_size; j++) {
      const auto& pj = avail_parts_[j];
      if (trained_interactions_[pi.idx][pj.idx] <= timestamp_) { done = false; break; }
      if (trained_interactions_[pj.idx][pi.idx] <= timestamp_) { done = false; break; }
    }
    // Policy: 1. Evict only 1 partition which is 'done': interacted with all in curr buffer
    // 2.Evict only when buffer is full
    if ( (avail_parts_size == capacity_) && (done && to_be_evicted.empty()) ) {
      to_be_evicted.push_back(pi);
    } else {
      to_be_retained.push_back(pi);
    }
  }
}

/*
Evict partition(s) based on a policy
1. If partitions with previous timestamps are present, evict them (one by one) --> currently
2. Else :(all partitions belong to the current epoch) -->  Evict partition(s) according to trained interactions based on a pluggable policy
*/
void WorkerNode::ProcessPartitions() {
  while (timestamp_ < num_epochs_) {
    std::vector<PartitionMetadata> partitions_done, avail_parts_replacement;
    {
      WriteLock w_lock(avail_parts_rw_mutex_);
      const int avail_parts_size = avail_parts_.size();
      SPDLOG_TRACE("Size of available partitions: {}", avail_parts_size);

      // Current policy only evicts when partition buffer is full
      bool buffer_full = (avail_parts_size == capacity_);
      
      // Find partition with timestamp less than worker_time_
      int old_partition_index = -1;
      for (int i = 0; buffer_full && i < avail_parts_size; i++) {
        if (avail_parts_[i].timestamp < timestamp_) {
          old_partition_index = i;
          break;
        }
      }

      if (old_partition_index != -1) {
        partitions_done.push_back(avail_parts_[old_partition_index]);
        avail_parts_replacement = avail_parts_;
        avail_parts_replacement.erase(avail_parts_replacement.begin()+old_partition_index);  
      } else {
        for (const auto& part: avail_parts_) assert(part.timestamp == timestamp_);
        updateProcessedPartitions();
        // Policy: evict partition(s)
        evictPartitions(partitions_done, avail_parts_replacement);
      }
      printPartitionChange(avail_parts_replacement);
    }

    // Update avail_parts.
    avail_parts_ = avail_parts_replacement;
    // TODO: Introduce a queue and put this into separate thread
    for (auto p : partitions_done) {
      SPDLOG_TRACE("Dispatching partition {} to co-ordinator..", p.idx);
      DispatchPartitionsToCoordinator(p);
    }
    // Increase sleep time to reduce contention for lock
    // TODO: Replace this by a condition variable.
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
  }
}