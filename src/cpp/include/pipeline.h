//
// Created by Jason Mohoney on 2/29/20.
//
#ifndef MARIUS_PIPELINE_H
#define MARIUS_PIPELINE_H

#include <time.h>

#include "batch.h"
#include "dataset.h"
#include "datatypes.h"
#include "model.h"

#define WAIT_TIME 100000 // 100 micro seconds
#define NANOSECOND 1
#define MICROSECOND 1000
#define MILLISECOND 1000000

// CPU Pipeline worker IDs
#define EMBEDDINGS_LOADER_ID 0
#define CPU_BATCH_PREP_ID 1
#define CPU_COMPUTE_ID 2
#define CPU_ACCUMULATE_ID 3
#define UPDATE_EMBEDDINGS_ID 4

// GPU Pipeline worker IDs
#define EMBEDDINGS_TRANSFER_ID 5
#define GPU_COMPUTE_ID 6
#define UPDATE_TRANSFER_ID 7

#define CPU_NUM_WORKER_TYPES 5
#define GPU_NUM_WORKER_TYPES 5

using std::atomic;
using std::thread;
using std::get;

enum class ThreadStatus {
    Running,
    WaitPush,
    WaitPop,
    Paused,
    Done
};

tuple<Indices, Indices> getIntersection(Indices ind1, Indices ind2, torch::DeviceType device);

torch::Tensor getMask(EdgeList all_edges, EdgeList batch_edges);

class Pipeline {
  public:
    DataSet *data_set_;
    Model *model_;

    int max_batches_in_flight_;
    atomic<int> batches_in_flight_;
    std::mutex *max_batches_lock_;
    std::condition_variable *max_batches_cv_;
    atomic<int64_t> edges_processed_;
    mutex timestamp_lock_;
    Timestamp oldest_timestamp_;
    std::mutex completed_batches_lock_;
    vector<Batch*> completed_batches_;
    std::shared_ptr<spdlog::logger> logger_;

    std::mutex *pipeline_lock_;
    std::condition_variable pipeline_cv_;

    struct timespec report_time_;
    thread monitor_;

    mutex ranks_lock_;
    torch::Tensor ranks_;

    atomic<int> admitted_batches_;

    bool train_;

  public:

    ~Pipeline();

    thread initThreadOfType(int worker_type, bool *paused, ThreadStatus *status, int device_id);

    virtual void addWorkersToPool(int pool_id, int worker_type, int num_workers) = 0;

    bool isDone();

    bool isTrain();

    void waitComplete();

    virtual void initialize() = 0;

    virtual void start() = 0;

    virtual void stopAndFlush() = 0;

    virtual void flushQueues() = 0;

    virtual void setQueueExpectingData(bool expecting_data) = 0;

    virtual void reportQueueStatus() = 0;

    virtual void reportThreadStatus() = 0;

    virtual size_t getCompletedBatchesSize() = 0;
    virtual void clearCompletedBatches() = 0;
    void reportMRR();
};

class PipelineCPU : public Pipeline {
  public:
    vector<thread> *pool_[CPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[CPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[CPU_NUM_WORKER_TYPES];
    vector<ThreadStatus> *trace_[CPU_NUM_WORKER_TYPES];

    Queue<Batch *> *loaded_batches_;
    Queue<Batch *> *prepped_batches_;
    Queue<Batch *> *unaccumulated_batches_;
    Queue<Batch *> *update_batches_;

    PipelineCPU(DataSet *data_set, Model *model, bool train, struct timespec report_time);

    ~PipelineCPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;

    void reportQueueStatus() override;

    void reportThreadStatus() override;

    size_t getCompletedBatchesSize() override {
      std::lock_guard<std::mutex> guard(completed_batches_lock_);
      return completed_batches_.size();
    }

    void clearCompletedBatches() override {
      std::lock_guard<std::mutex> guard(completed_batches_lock_);
      completed_batches_.clear();
    }
};

class PipelineGPU : public Pipeline {
  public:
    vector<thread> *pool_[GPU_NUM_WORKER_TYPES];
    vector<bool *> *pool_paused_[GPU_NUM_WORKER_TYPES];
    vector<ThreadStatus *> *pool_status_[GPU_NUM_WORKER_TYPES];
    vector<ThreadStatus> *trace_[GPU_NUM_WORKER_TYPES];

    Queue<Batch *> *loaded_batches_;
    std::vector<Queue<Batch *> *> device_loaded_batches_; // multiple queues for multiple devices
    std::vector<Queue<Batch *> *> device_update_batches_; // multiple queues for multiple devices
    Queue<Batch *> *update_batches_;

    PipelineGPU(DataSet *data_set, Model *model, bool train, struct timespec report_time);

    ~PipelineGPU();

    void addWorkersToPool(int pool_id, int worker_type, int num_workers) override;

    void initialize() override;

    void start() override;

    void stopAndFlush() override;

    void flushQueues() override;

    void setQueueExpectingData(bool expecting_data) override;

    void reportQueueStatus() override;

    void reportThreadStatus() override;

    size_t getCompletedBatchesSize() override {
      std::lock_guard<std::mutex> guard(completed_batches_lock_);
      return completed_batches_.size();
    }
    
    void clearCompletedBatches() override {
      std::lock_guard<std::mutex> guard(completed_batches_lock_);
      completed_batches_.clear();
    }
};

class PipelineMonitor {
  private:
    Pipeline *pipeline_;
    struct timespec sleep_time_;
  public:
    PipelineMonitor(Pipeline *pipeline, struct timespec sleep_time);

    void run();

};

class Worker {
  protected:
    Pipeline *pipeline_;
    struct timespec sleep_time_;
    bool *paused_;
    ThreadStatus *status_;

  public:
    Worker(Pipeline *pipeline, bool *paused, ThreadStatus *status);

    virtual void run() = 0;
};

class LoadEmbeddingsWorker : protected Worker {
  public:
    LoadEmbeddingsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class EmbeddingsToDeviceWorker : protected Worker {
  public:
    EmbeddingsToDeviceWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class PrepareBatchWorker : protected Worker {
  private:
  public:
    PrepareBatchWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class ComputeWorkerCPU : protected Worker {
  private:
  public:
    ComputeWorkerCPU(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {}

    void run() override;
};

class ComputeWorkerGPU : protected Worker {
  private:
    int device_id_;
  public:
    ComputeWorkerGPU(Pipeline *pipeline, int device_id, bool *paused, ThreadStatus *status);

    void run() override;
};

class AccumulateGradientsWorker : protected Worker {
  public:
    AccumulateGradientsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

class GradientsToHostWorker : protected Worker {
  private:
    int device_id_;
  public:
    GradientsToHostWorker(Pipeline *pipeline, int device_id, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status}, device_id_{device_id} {};

    void run() override;
};

class UpdateEmbeddingsWorker : protected Worker {
  public:
    UpdateEmbeddingsWorker(Pipeline *pipeline, bool *paused, ThreadStatus *status) : Worker{pipeline, paused, status} {};

    void run() override;
};

#endif //MARIUS_PIPELINE_H


