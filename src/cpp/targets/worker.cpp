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
#include <experimental/filesystem>


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
namespace fs = std::experimental::filesystem;


///



//
// TODO List:
// 1. Process group (pg_) is not thread safe, requires locking.
// 2. Multiple process groups are needed, atleast one for 
// metadata communication and one for bulk transfer.
// 3. Interactions need to be transferred to coordinator.
class WorkerNode {
  public:
    explicit WorkerNode(
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
        this->TransferPartitionsToWorkerNodes();
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
    PartitionMetadata RequestPartition() {
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
      return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
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
          PartitionMetadata p = RequestPartition();
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
    bool sendPartition(const PartitionMetadata& part, int dstRank) {
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
          PartitionMetadata part = dispatch_parts_.front();
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
    void TransferPartitionsFromWorkerNodes() {
      while (1) {
        {
          while (1) {
            WriteLock w_lock(transfer_receive_parts_rw_mutex_);
            if (transfer_receive_parts_.empty()) break;
            PartitionMetadata part  = transfer_receive_parts_.front();
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
    void TransferPartitionsToWorkerNodes() {
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
        // std::vector<at::Tensor> tensors;
        // tensors.push_back(std::move(*node_map[part_idx]));
        // pg_->send();



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
  std::vector<PartitionMetadata> avail_parts_;
  mutable std::shared_mutex dispatch_parts_rw_mutex_;
  std::queue<PartitionMetadata> dispatch_parts_;
  mutable std::shared_mutex transfer_receive_parts_rw_mutex_;
  std::queue<PartitionMetadata> transfer_receive_parts_;
  std::vector<std::thread> threads_;
  mutable std::shared_mutex node_map_rw_mutex_;
  std::vector<std::shared_ptr<torch::Tensor>> node_map;
  int partition_size_;
  int embedding_dims_;
};


int main(int argc, char* argv[]) {
  marius_options = parseConfig(argc, argv); // marius options is an extern variable form config.h that is globally used across the library.
  int rank = marius_options.communication.rank;
  int world_size = marius_options.communication.world_size;
  std::string prefix = marius_options.communication.prefix;
  std::cout << "Rank : " << rank << ", " << "World size: " << world_size << ", " << "Prefix: " << prefix << std::endl;
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
  WorkerNode worker(pg, rank, capacity, num_partitions, world_size-1);
  std::string log_file = marius_options.general.experiment_name;
  MariusLogger marius_logger = MariusLogger(log_file);
  spdlog::set_default_logger(marius_logger.main_logger_);
  marius_logger.setConsoleLogLevel(marius_options.reporting.log_level);
  bool gpu = false;
  if (marius_options.general.device == torch::kCUDA) {
      gpu = true;
  }
  Timer preprocessing_timer = Timer(gpu);
  preprocessing_timer.start();
  SPDLOG_INFO("Start preprocessing");

  DataSet *train_set;
  DataSet *eval_set;

  Model *model = initializeModel(marius_options.model.encoder_model, marius_options.model.decoder_model);

  bool train = true;

  if (train) {
      tuple<Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *, Storage *> storage_ptrs = initializeTrain();
      Storage *train_edges = get<0>(storage_ptrs);
      Storage *eval_edges = get<1>(storage_ptrs);
      Storage *test_edges = get<2>(storage_ptrs);

      Storage *embeddings = get<3>(storage_ptrs);
      Storage *emb_state = get<4>(storage_ptrs);

      Storage *src_rel = get<5>(storage_ptrs);
      Storage *src_rel_state = get<6>(storage_ptrs);
      Storage *dst_rel = get<7>(storage_ptrs);
      Storage *dst_rel_state = get<8>(storage_ptrs);

      bool will_evaluate = !(marius_options.path.validation_edges.empty() && marius_options.path.test_edges.empty());

      train_set = new DataSet(train_edges, embeddings, emb_state, src_rel, src_rel_state, dst_rel, dst_rel_state);
      SPDLOG_INFO("Training set initialized");
      if (will_evaluate) {
          eval_set = new DataSet(train_edges, eval_edges, test_edges, embeddings, src_rel, dst_rel);
          SPDLOG_INFO("Evaluation set initialized");
      }

      preprocessing_timer.stop();
      int64_t preprocessing_time = preprocessing_timer.getDuration();

      SPDLOG_INFO("Preprocessing Complete: {}s", (double) preprocessing_time / 1000);

      Trainer *trainer;
      Evaluator *evaluator;

      if (marius_options.training.synchronous) {
          trainer = new SynchronousTrainer(train_set, model);
      } else {
          trainer = new PipelineTrainer(train_set, model);
      }

      if (will_evaluate) {
          if (marius_options.evaluation.synchronous) {
              evaluator = new SynchronousEvaluator(eval_set, model);
          } else {
              evaluator = new PipelineEvaluator(eval_set, model);
          }
      }

      for (int epoch = 0; epoch < marius_options.training.num_epochs; epoch += marius_options.evaluation.epochs_per_eval) {
          int num_epochs = marius_options.evaluation.epochs_per_eval;
          if (marius_options.training.num_epochs < num_epochs) {
              num_epochs = marius_options.training.num_epochs;
              trainer->train(num_epochs);
          } else {
              trainer->train(num_epochs);
              if (will_evaluate) {
                  evaluator->evaluate(epoch + marius_options.evaluation.epochs_per_eval < marius_options.training.num_epochs);
              }
          }
      }
      embeddings->unload(true);
      src_rel->unload(true);
      dst_rel->unload(true);


      // garbage collect
      delete trainer;
      delete train_set;
      if (will_evaluate) {
          delete evaluator;
          delete eval_set;
      }

      freeTrainStorage(train_edges, eval_edges, test_edges, embeddings, emb_state, src_rel, src_rel_state, dst_rel, dst_rel_state);

  }
  worker.start_working();
  worker.stop_working();
}