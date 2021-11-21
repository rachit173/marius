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
  PartitionMetadata(int idx_, int src_, int num_partitions): idx(idx_), src(src_), interactions(num_partitions, 0) {}
  void clear_interactions() {
    for (auto& i : interactions) i = 0;
  }
  int idx;
  int src;
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
    torch::Tensor tensor = torch::zeros({n+3});
    tensor.data_ptr<float>()[0] = idx;
    tensor.data_ptr<float>()[1] = src;
    tensor.data_ptr<float>()[2] = n;
    for (int i = 0; i < n; i++) {
      if (interactions[i]) tensor.data_ptr<float>()[i+3] = 1;
      else tensor.data_ptr<float>()[i+3] = 0;
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
    int n = tensor.data_ptr<float>()[2];
    auto p = PartitionMetadata(idx, src, n);
    for (int i = 0; i < n; i++) {
      if (tensor.data_ptr<float>()[i+3]==1) p.interactions[i] = 1;
    }
    return p; 
  }
};