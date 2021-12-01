
class CoordinatorTagGenerator {
  public:
  CoordinatorTagGenerator(int num_workers): num_workers_(num_workers), self_rank_(num_workers) {
    offset1_ = num_workers + 1;
    offset2_ = offset1_ + num_workers;
    offset3_ = offset2_ + num_workers * num_workers;
  }
  int getCoordinatorCommandTag() {
    return 0;
  }
  int getWorkerSpecificCommunicationTag(int worker_rank) {
    return 1 + worker_rank;
  }
  int getWorkerSpecificEpochSignalingTag(int worker_rank) {
    return offset3_ +  worker_rank;
  }
  private:
  int self_rank_;
  int num_workers_;
  int offset1_;
  int offset2_;
  int offset3_;
};


class WorkerTagGenerator {
  public:
  WorkerTagGenerator(int self_rank, int num_workers): self_rank_(self_rank), num_workers_(num_workers) {
    offset1_ = num_workers + 1;
    offset2_ = offset1_ + num_workers;
    offset3_ = offset2_ + num_workers * num_workers;
  }
  int getCoordinatorCommandTag() {
    return 0;
  }
  int getCoordinatorSpecificCommunicationTag() {
    return 1 + self_rank_;
  }
  int getTagWhenRequesterCommandPath(int dst_rank) {
    return offset1_+dst_rank;
  }
  int getTagWhenRequesterDataPath(int dst_rank) {
    return offset2_+(self_rank_*num_workers_+dst_rank);
  }
  int getTagWhenReceiverCommandPath() {
    return offset1_+self_rank_;
  }
  int getTagWhenReceiverDataPath(int dst_rank) {
    return offset2_+(dst_rank*num_workers_+self_rank_);
  }
  int getEpochSignalingTag() {
    return offset3_ +  self_rank_;
  }
  private:
  int self_rank_;
  int num_workers_;
  int offset1_;
  int offset2_;
  int offset3_;
};