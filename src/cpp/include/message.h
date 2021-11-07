#include <torch/torch.h>
#include <vector>

struct Request {
  int type;
};

struct Response {
  int partition_num;
  int rank;
};


struct PartitionMetadata {
  PartitionMetadata(int idx_): idx(idx_), src(-1) {}
  PartitionMetadata(int idx_, int src_): idx(idx_), src(src_) {}
  PartitionMetadata(int idx_, int src_, int num_partitions): idx(idx_), src(src_) {
    interactions.resize(num_partitions, 0);
  }
  void clear_interactions() {
    for (auto& i : interactions) i = 0;
  }
  int idx;
  int src;
  std::vector<int> interactions;
  torch::Tensor ConvertToTensor() const {
    torch::Tensor tensor = torch::zeros({2});
    tensor.data_ptr<float>()[0] = idx;
    tensor.data_ptr<float>()[1] = src;
    return tensor;
  }
  static PartitionMetadata ConvertToPartition(const torch::Tensor& tensor) {
    int idx = tensor.data_ptr<float>()[0];
    int src = tensor.data_ptr<float>()[1];
    return PartitionMetadata(idx, src);
  }
};