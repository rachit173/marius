#ifndef WORKER_H
#define WORKER_H

#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"
#include "communication.h"

#include <algorithm>
#include <vector>
#include <shared_mutex>
#include <mutex>

#include <torch/torch.h>
#include <c10d/ProcessGroupGloo.hpp>

typedef std::shared_mutex RwLock;
typedef std::unique_lock<RwLock> WriteLock;
typedef std::shared_lock<RwLock> ReadLock;

class WorkerNode {
private:
	// Fields used for communication
	std::shared_ptr<c10d::ProcessGroupGloo> pg_;
	int rank_;
	void* send_buffer_;
	void* receive_buffer_;

	// Configuration for distributed setup
	int num_workers_;
	int capacity_;
	int coordinator_rank_;
	int sleep_time_;

	// Shared data structure & mutexes
	mutable std::shared_mutex avail_parts_rw_mutex_;
	std::vector<PartitionMetadata> avail_parts_;
	mutable std::mutex next_epoch_mutex_;
	std::vector<std::thread> threads_;

	// Progress-tracking structures
	vector<vector<int>> processed_interactions_;
	vector<vector<int>> trained_interactions_;

	// Fields from Marius pipeline
	int num_partitions_;
	int partition_size_;
	int dtype_size_;
	int embedding_size_; 
	int embedding_dims_;
	int timestamp_;
	int num_epochs_;
	bool gpu_;
	DataSet* trainset_;
	DataSet* evalset_;
	Trainer* trainer_;
	Evaluator* evaluator_;
	Pipeline* pipeline_;
	PartitionBufferStorage* embeds_;
	PartitionBufferStorage* embeds_state_;
	PartitionBuffer* pb_embeds_;
	PartitionBuffer* pb_embeds_state_;
	WorkerTagGenerator tag_generator_;
	std::string perf_metrics_label_;

	

	// Partition communication
	void TransferPartitionsFromWorkerNodes(PartitionMetadata part);
	void receivePartition(int idx, int src);
	void receivePartitionToBuffer(PartitionBuffer *partition_buffer, int src);
	void TransferPartitionsToWorkerNodes();
	void sendPartitionFromBuffer(PartitionBuffer* partition_buffer, int src_rank, int partition_index);
	void forceToBuffer(PartitionBuffer *partition_buffer, int partition_idx);
	
	bool sendPartition(PartitionMetadata part, int dstRank);
	PartitionMetadata receivePartition(int srcRank);

	// Util
	// Process new partition
	void ClearProcessedInteractions();
	int getSize();
	void ProcessNewPartition(PartitionMetadata p);

	// Dispatch done partition
	void DispatchPartitionsToCoordinator(PartitionMetadata part);
	void printPartitionChange(vector<PartitionMetadata> &avail_parts_replacement);
	void updateProcessedPartitions();

	// Flush partitions to disk for Evaluation
	void flushPartitions(PartitionBuffer* partition_buffer);

	
	// Worker Policy:
	// Order of executing batches
	virtual void orderInteractions(vector<pair<int, int>>& interactions);
	// Selecting partition(s) to evict
	virtual void evictPartitions(vector<PartitionMetadata> &to_be_evicted,
								 vector<PartitionMetadata> &to_be_retained);

public:
    explicit WorkerNode(
      std::shared_ptr<c10d::ProcessGroupGloo> pg,
      int rank,
      int capacity,
      int num_partitions,
      int num_workers,
      int num_epochs,
      bool gpu
    ):
    pg_(pg), 
    rank_(rank),
    capacity_(capacity),
    num_partitions_(num_partitions),
    num_workers_(num_workers),
    tag_generator_(rank, num_workers),
    num_epochs_(num_epochs),
    timestamp_(0),
    gpu_(gpu) {
      coordinator_rank_ = num_workers_; 
      sleep_time_ = 500; // us
      processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      trained_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
      perf_metrics_label_ = "[Performance Metrics]";
    }

	void start_working(DataSet *trainset, DataSet *evalset, Trainer *trainer, Evaluator *evaluator, PartitionBufferStorage *embeds, PartitionBufferStorage *embeds_state);
	void stop_working() {}

	// Thread main methods
	void RequestPartitions();

	// Worker Policy: dispatching batch/batches to the co-ordinator
	void ProcessPartitions();
	
	void PrepareForNextEpoch();
	void ServiceEvaluationRequest();
	
	void RunTrainer();
};



int worker_main(int argc, char *argv[]);

#endif