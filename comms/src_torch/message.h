
struct Request {
  int type;
};

struct Response {
  int partition_num;
  int rank;
};


struct Partition {
  Partition(int idx_): idx(idx_), src(-1) {}
  Partition(int idx_, int src_): idx(idx_), src(src_) {}
  int idx;
  int src;
};