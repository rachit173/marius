#include "coordinator.h"
#include "communication.h"
#include "message.h"
#include <torch/torch.h>


// Allocate partition
PartitionMetadata Coordinator::AllocatePartition(int srcRank) {
  // Assumes that total # of partitions >=  # of workers * Buffer capacity per worker
  if(available_partitions_.empty()) {
    // Not available
    SPDLOG_TRACE("Partitions not available for worker with rank: {}", srcRank);
    return PartitionMetadata(-1, -1, timestamp_, num_partitions_);
  }

  // Dispatch partition which can do maximum interactions considering the existing state of worker and already done interactions
  // and take the maximum
  vector<int> possible_interactions(num_partitions_, 0);
  int max_interactions = 0;
  PartitionMetadata response_partition(-1, -1, timestamp_, num_partitions_);
  for(const auto& p: available_partitions_){
    for(int j = 0; j < num_partitions_; j++){
      if(j == p.idx){
        possible_interactions[p.idx] += (processed_interactions_[p.idx][j] <= timestamp_);
      } else if(in_process_partitions_[srcRank][j] == 1){
        // if not processed, add as an interaction. Can add symmetric interaction to double, but no use....
        possible_interactions[p.idx] += 2 * (processed_interactions_[p.idx][j] <= timestamp_);
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
}

// Sync co-ordinator state after receiving partition
void Coordinator::syncPartitionMetadata(PartitionMetadata& part){
  // update co-ordinator view of partitions and interactions
  available_partitions_.push_back(part);
  in_process_partitions_[part.src][part.idx] = 0;
  for(int i = 0; i < num_partitions_; i++){
    processed_interactions_[part.idx][i] = std::max(processed_interactions_[part.idx][i], part.interactions[i]);
    processed_interactions_[i][part.idx] = std::max(processed_interactions_[i][part.idx], part.interactions[i]);
  }
}

