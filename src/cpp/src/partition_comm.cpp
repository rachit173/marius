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

// ##################### Receive Partition from another worker ####################

void WorkerNode::TransferPartitionsFromWorkerNodes(PartitionMetadata part) {
    if (part.src == -1 || part.src == rank_) {
        // Already have the partition.
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Partition Not Requested\", 'data':('partition index':{}, 'src':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx, part.src);
    } else {
        receivePartition(part.idx, part.src);
    }

    {
        forceToBuffer(pb_embeds_, part.idx);
        forceToBuffer(pb_embeds_state_, part.idx);
        SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Forced to Buffer\", 'data':('partition index':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part.idx);
    }

    // Add the received partitions to avail_parts_ vector.
    {
        WriteLock w_lock(avail_parts_rw_mutex_);
        SPDLOG_TRACE("Pushed to avail parts: {}", part.idx);
        avail_parts_.push_back(part);
    }
}

// TODO: Analyse + Debug
// Force partition to Partition Buffer as soon as it is received
void WorkerNode::forceToBuffer(PartitionBuffer *partition_buffer, int partition_idx){
    // Admit partition into partition buffer forcefully
    Partition *partition = partition_buffer->getPartitionTable()[partition_idx];
    partition_buffer->admitWithLock(partition);
}

// Receive partition (embeddings + optimizer state) from another worker and write it to disk
void WorkerNode::receivePartition(int idx, int src) {
    // TODO : add lock to this method
    // Ask the other worker for the partition.
    torch::Tensor request = torch::zeros({1}) + idx;
    std::vector<at::Tensor> request_tensors({request});
    auto send_work = pg_->send(request_tensors, src, tag_generator_.getTagWhenRequesterCommandPath(src));
    if (send_work) {
        send_work->wait();
    }
    
    Timer timer = Timer(gpu_);
    timer.start();
    
    // Receive Partition for node embeddings and optimizer state
    receivePartitionToBuffer(pb_embeds_, src);
    receivePartitionToBuffer(pb_embeds_state_, src);
    
    timer.stop();
    int64_t event_time = timer.getDuration();
    SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Received Partition\", 'data':('partition index':{}, 'src':{}, 'dest': {}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, rank_, idx, src, rank_, event_time);
}

void WorkerNode::receivePartitionToBuffer(PartitionBuffer *partition_buffer, int src) {
    // Receive metadata
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor node_embed_tensor = torch::zeros({1, 5}, options);
    std::vector<torch::Tensor> node_embed_tensors({node_embed_tensor});
    auto recv_work = pg_->recv(node_embed_tensors, src, tag_generator_.getTagWhenRequesterDataPath(src));
    if (recv_work) {
        recv_work->wait();
    }
    // Create Partition from metadata received
    auto partition = Partition::ConvertToPartition(node_embed_tensors[0]);

    // Process: 
    // 1. Allocate memory for the receiving the incoming partition (posix_memalign)
    // 2. Receive data into tensor, which points under the hood to the same allocated memory
    // 3. Write the recvd partition to disk(partition file)
    
    // Receive partition data
    options = torch::TensorOptions().dtype(partition->dtype_);
    // 1. Point data_ptr_ to the receive buffer
    partition->data_ptr_ = receive_buffer_;
    
    torch::Tensor tensor_data_recvd = torch::from_blob(receive_buffer_, {partition->partition_size_,partition->embedding_size_},partition->dtype_);              
    // TODO: [Optimization]: class Single large space, any size of partition can be copied there
    // then directly use pwrite to copy to correct portion of node embeddings
    
    // 2. Receive the tensor having data from the worker.
    std::vector<torch::Tensor> tensors_data_recvd({tensor_data_recvd});
    auto recv_data_work = pg_->recv(tensors_data_recvd, src, tag_generator_.getTagWhenRequesterDataPath(src));
    if (recv_data_work) {
        recv_data_work->wait();
    }

    // 3. Write fetched partition to partitioned file.
    std::vector<Partition *>& partition_table = partition_buffer->getPartitionTable();
    PartitionedFile *partition_file = partition_buffer->getPartitionedFile();
    partition_file->writePartition(partition.get());

    partition->data_ptr_ = nullptr;
}

// #################################################################################

// ######################## Send Partition to another worker ###########################
void WorkerNode::TransferPartitionsToWorkerNodes() {
    // Receive the request and transfer the partition from the node map
    while (timestamp_ < num_epochs_) {
    // Receive request for transfer
    torch::Tensor request = torch::zeros({1});
    std::vector<at::Tensor> tensors({request});
    auto recv_work = pg_->recvAnysource(tensors, tag_generator_.getTagWhenReceiverCommandPath());
    int src_rank;
    if (recv_work) {
        recv_work->wait();
        src_rank = recv_work->sourceRank();
    }
    // send partition metadata
    int part_idx = tensors[0].data_ptr<float>()[0];

    Timer timer = Timer(gpu_);
    timer.start();

    // Send Partition for node embeddings and optimizer state
    sendPartitionFromBuffer(pb_embeds_, src_rank, part_idx);
    sendPartitionFromBuffer(pb_embeds_state_, src_rank, part_idx);

    timer.stop();
    int64_t event_time = timer.getDuration();
    SPDLOG_DEBUG("{} ('epoch': {}, 'worker': {}, 'event':\"Sent Partition\", 'data':('partition index':{}, 'src': {}, 'dest':{}, 'duration':{}))", perf_metrics_label_, timestamp_ + 1, rank_, part_idx, rank_, src_rank, event_time);
    }
}

void WorkerNode::sendPartitionFromBuffer(PartitionBuffer* partition_buffer, int src_rank, int partition_index){
    std::vector<Partition*>& partition_table = partition_buffer->getPartitionTable();
    PartitionedFile* partition_file = partition_buffer->getPartitionedFile();
    // metadata: partition_id_, partition_size_, embedding_size_, idx_offset_, file_offset_
    torch::Tensor partition_metadata = partition_table[partition_index]->ConvertMetaDataToTensor();
    std::vector<torch::Tensor> tensors_to_send({partition_metadata});
    // send metadata
    auto send_serialized_partition = pg_->send(tensors_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
    if (send_serialized_partition) {
        send_serialized_partition->wait();
    }                
    // Send partition data:
    // 1. Read partition from disk/partition buffer
    // 2. Convert to tensor and send

    Partition *partition_to_be_sent = partition_table[partition_index];
    torch::Tensor tensor_data_to_send;

    // Directly send partition from partition buffer if present in the buffer
    try {
        if(partition_to_be_sent->present_){
            std::lock_guard<std::mutex> guard(partition_buffer->getAdmitLock());
            tensor_data_to_send = partition_to_be_sent->ConvertDataToTensor();

            std::vector<torch::Tensor> tensors_data_to_send({tensor_data_to_send});
            // send partition data
            auto send_part_data = pg_->send(tensors_data_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
            if (send_part_data){
                send_part_data->wait();
            }
            return;
        }
    } catch(const std::exception& e){
        std::cout << e.what() << std::endl;
    }
    // If not present in buffer: read from disk and send partition data
    //else
    {
        partition_file->readPartition(send_buffer_, partition_to_be_sent);
        tensor_data_to_send = partition_table[partition_index]->ConvertDataToTensor();
        std::vector<torch::Tensor> tensors_data_to_send({tensor_data_to_send});
        // send partition data
        auto send_part_data = pg_->send(tensors_data_to_send, src_rank, tag_generator_.getTagWhenReceiverDataPath(src_rank));
        if (send_part_data){
            send_part_data->wait();
        }
    }
}
// #################################################################################


// ##############Communication code#################

// TODO: Move to PartitionMetadata
PartitionMetadata WorkerNode::receivePartition(int srcRank) {
    torch::Tensor tensor = torch::zeros({1}) + 1; // command is 1 for request partition.
    std::vector<at::Tensor> tensors({tensor});
    auto send_work = pg_->send(tensors, srcRank, tag_generator_.getCoordinatorCommandTag());
    if (send_work) {
        send_work->wait();
    }
    SPDLOG_TRACE("Started Receiving");
    // Receive the returned partition
    torch::Tensor part_tensor = torch::zeros({num_partitions_+kPartititionMetadataSerde});
    std::vector<at::Tensor> part_tensor_vec({part_tensor});
    auto recv_work = pg_->recv(part_tensor_vec, srcRank, tag_generator_.getCoordinatorSpecificCommunicationTag());
    if (recv_work) {
       recv_work->wait();
    }
    return PartitionMetadata::ConvertToPartition(part_tensor_vec[0]);
}

// TODO: Move to PartitionMetadata
bool WorkerNode::sendPartition(PartitionMetadata part, int dstRank) {
    torch::Tensor tensor = part.ConvertToTensor();
    std::vector<at::Tensor> tensors({tensor});
    auto send_work = pg_->send(tensors, dstRank, tag_generator_.getCoordinatorSpecificCommunicationTag());
    if (send_work) {
        send_work->wait();
    } else {
        return false;
    }
    return true;
}
// #################################################################################