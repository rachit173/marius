/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

#include <iostream>
#include <memory>
#include <chrono>

#include "gloo/allreduce_ring.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/tcp/device.h"
#include "gloo/transport/tcp/unbound_buffer.h"
#include "gloo/transport/tcp/context.h"
#include "message.h"
// TODO(rrt): Add usage.

using gloo::transport::tcp::UnboundBuffer;

class Coordinator {
  struct Partition {
    Partition(int i): idx(i) {}
    int idx;
  };
  public:
    explicit Coordinator(
      std::shared_ptr<gloo::transport::tcp::Context> context,
std::shared_ptr<gloo::rendezvous::Context> rendezvous_context,
    int num_partitions,
    int num_machines):
    context_(context),
    rendezvous_context_(rendezvous_context),
    num_partitions_(num_partitions),
    num_machines_(num_machines) {
      // Setup internal data structures.



    }
    void start_listening() {
      const int size = 1024;
      char buff[size];
      auto receiver = context_->createUnboundBuffer(buff, size);
      // auto receiver = std::make_unique<UnboundBuffer>(context_, buff, size);
      int sender_rank;
      std::chrono::milliseconds timeout(2000);
      std::vector<int> srcRanks;
      for (int i = 0; i <= num_machines_; i++) { srcRanks.push_back(i); }
      auto slot = rendezvous_context_->nextSlot();
      while(1) {
        receiver->recv(srcRanks, slot);
        // receiver->recv()
        if (receiver->waitRecv(&sender_rank, timeout)) {
          std::cout << "Received: " << buff <<", from: " << sender_rank << "\n";
        } else {
          std::cerr << "Aborted receiving" << "\n";
        }
      }
    }
    void stop_listening() {

    }
  private:
    std::shared_ptr<gloo::transport::tcp::Context> context_;
    int num_partitions_;
    int num_machines_;
    std::shared_ptr<gloo::rendezvous::Context> rendezvous_context_;
    std::vector<Partition> available_partitions_;
};

// Communication Protocol
// rank == size-1 is the coordinator
// If there are 5 processes with rank [0, 1, 2, 3, 4],
// the process with rank 4 is the coordinator.
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "The command should be run as ./a.aout <prefix> <rank> <size>" << std::endl;
    return 0;
  }
  // The following statement creates a TCP "device" for Gloo to use.
  // See "gloo/transport/device.h" for more information. For the
  // purposes of this example, it is sufficient to see the device as
  // a factory for every communication pair.
  //
  // The argument to gloo::transport::tcp::CreateDevice is used to
  // find the network interface to bind connection to. The attr struct
  // can be populated to specify exactly which interface should be
  // used, as shown below. This is useful if you have identical
  // multi-homed machines that all share the same network interface
  // name, for example.
  //
  gloo::transport::tcp::attr attr;
  //attr.iface = "eth0";
  //attr.iface = "ib0";
  attr.iface = "lo";

  // attr.ai_family = AF_INET; // Force IPv4
  // attr.ai_family = AF_INET6; // Force IPv6
  attr.ai_family = AF_UNSPEC; // Use either (default)

  // A string is implicitly converted to an "attr" struct with its
  // hostname field populated. This will try to resolve the interface
  // to use by resolving the hostname or IP address, and finding the
  // corresponding network interface.
  //
  // Hostname "localhost" should resolve to 127.0.0.1, so using this
  // implies that all connections will be local. This can be useful
  // for single machine operation.
  //
  //   auto dev = gloo::transport::tcp::CreateDevice("localhost");
  //

  auto dev = gloo::transport::tcp::CreateDevice(attr);

  // Now that we have a device, we can connect all participating
  // processes. We call this process "rendezvous". It can be performed
  // using a shared filesystem, a Redis instance, or something else by
  // extending it yourself.
  //
  // See "gloo/rendezvous/store.h" for the functionality you need to
  // implement to create your own store for performing rendezvous.
  //
  // Below, we instantiate rendezvous using the filesystem, given that
  // this example uses multiple processes on a single machine.
  //
  auto fileStore = gloo::rendezvous::FileStore("./rendezvous_point/");

  // To be able to reuse the same store over and over again and not have
  // interference between runs, we scope it to a unique prefix with the
  // PrefixStore. This wraps another store and prefixes every key before
  // forwarding the call to the underlying store.
  std::string prefix(argv[1]);
  std::cout << "Prefix " << prefix << std::endl;
  auto prefixStore = gloo::rendezvous::PrefixStore(prefix, fileStore);

  // Using this store, we can now create a Gloo context. The context
  // holds a reference to every communication pair involving this
  // process. It is used by every collective algorithm to find the
  // current process's rank in the collective, the collective size,
  // and setup of send/receive buffer pairs.
  const int rank = atoi(argv[2]);
  const int size = atoi(argv[3]);
  std::cout << "My rank: " << rank << std::endl;
  auto rendezvous_context = std::make_shared<gloo::rendezvous::Context>(rank, size);

  rendezvous_context->connectFullMesh(prefixStore, dev);
  int num_partitions = 10;
  int world_size = size-1;
  auto dev_clone = dev;
  auto t = std::dynamic_pointer_cast<gloo::transport::tcp::Device>(std::move(dev_clone));
  auto context = std::make_shared<gloo::transport::tcp::Context>(t, rank, size);
  Coordinator coordinator(context, rendezvous_context, num_partitions, world_size);
  coordinator.start_listening();
  coordinator.stop_listening();
  // All connections are now established. We can now initialize some
  // test data, instantiate the collective algorithm, and run it.
  // std::array<int, 4> data;
  // std::cout << "Input: " << std::endl;
  // for (int i = 0; i < data.size(); i++) {
  //   data[i] = i;
  //   std::cout << "data[" << i << "] = " << data[i] << std::endl;
  // }

  // Allreduce operates on memory that is already managed elsewhere.
  // Every instance can take multiple pointers and perform reduction
  // across local buffers as well. If you have a single buffer only,
  // you must pass a std::vector with a single pointer.
  // std::vector<int*> ptrs;
  // ptrs.push_back(&data[0]);

  // The number of elements at the specified pointer.
  // int count = data.size();

  // Instantiate the collective algorithm.
  // auto allreduce =
  //   std::make_shared<gloo::AllreduceRing<int>>(
  //     context, ptrs, count);

  // Run the algorithm.
  // allreduce->run();

  // Print the result.
  // std::cout << "Output: " << std::endl;
  // for (int i = 0; i < data.size(); i++) {
  //   std::cout << "data[" << i << "] = " << data[i] << std::endl;
  // }

  return 0;
}