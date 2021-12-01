#pragma once
#include <torch/torch.h>
#include <vector>


struct Request {
  int type;
};

struct Response {
  int partition_num;
  int rank;
};

static const int kPartititionMetadataSerde = 4;

struct PartitionMetadata {
  PartitionMetadata(int idx_, int src_, int timestamp_, int num_partitions): 
  idx(idx_), src(src_), 
  timestamp(timestamp_),
  interactions(num_partitions, 0) {}
  void updateTimestamp(int ts) {
    timestamp = ts;
    for (int i = 0; i < interactions.size(); i++) {
      interactions[i] = ts;
    }
  }
  int idx;
  int src;
  int timestamp;
  std::vector<int> interactions;
  /**
   * @brief 
   * The tensor T comprises of the format,
   * T[0] = partition id
   * T[1] = partition containing worker id (-1 for no particular source worker)
   * T[2] = number of partitions
   * T[k], k=3..n-1 = boolean indicating interaction (self_id, T[k]) has been performed
   * 
   * @return torch::Tensor 
   */
  torch::Tensor ConvertToTensor() const {
    int n = interactions.size();
    torch::Tensor tensor = torch::zeros({n+kPartititionMetadataSerde});
    tensor.data_ptr<float>()[0] = idx;
    tensor.data_ptr<float>()[1] = src;
    tensor.data_ptr<float>()[2] = timestamp;
    tensor.data_ptr<float>()[3] = n;
    for (int i = 0; i < n; i++) {
      if (interactions[i]) tensor.data_ptr<float>()[i+kPartititionMetadataSerde] = 1;
      else tensor.data_ptr<float>()[i+kPartititionMetadataSerde] = 0;
    }
    return tensor;
  }
  /**
   * @brief 
   * 
   * @param tensor For conversion to a partition 
   * @return PartitionMetadata 
   */
  static PartitionMetadata ConvertToPartition(const torch::Tensor& tensor) {
    int idx = tensor.data_ptr<float>()[0];
    int src = tensor.data_ptr<float>()[1];
    int timestamp = tensor.data_ptr<float>()[2];
    int n = tensor.data_ptr<float>()[3];
    auto p = PartitionMetadata(idx, src, timestamp, n);
    for (int i = 0; i < n; i++) {
      if (tensor.data_ptr<float>()[i+kPartititionMetadataSerde]==1) p.interactions[i] = 1;
    }
    return p; 
  }
};