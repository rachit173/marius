#ifndef COORD_H
#define COORD_H

#include "message.h"
#include "config.h"
#include "evaluator.h"
#include "io.h"
#include "logger.h"
#include "model.h"
#include "trainer.h"
#include "util.h"
#include "communication.h"

#include <iostream>
#include <vector>
#include <memory>
#include <exception>
#include <filesystem>
#include <algorithm>

#include <torch/torch.h>
#include <c10d/TCPStore.hpp>
#include <c10d/PrefixStore.hpp>
#include <c10d/Store.hpp>
#include <c10d/ProcessGroupGloo.hpp>
#include <c10d/GlooDeviceFactory.hpp>
#include <c10d/frontend.hpp>

class Coordinator
{
private:
    std::shared_ptr<c10d::ProcessGroupGloo> pg_;
    const int num_partitions_;
    const int num_workers_;
    const int num_epochs_;
    const int epochs_per_eval_;
    std::vector<PartitionMetadata> available_partitions_;
    std::vector<vector<int>> in_process_partitions_;
    std::vector<vector<int>> processed_interactions_;
    CoordinatorTagGenerator tag_generator_;
    int timestamp_;
    std::string perf_metrics_label_;
    Timer epoch_timer_;

    //####### Handle Epoch completion ##########
    double getCompletionRatio(int ts);
    void updateInteractionsTimestamp(int ts);
    void updateAvailablePartitionsTimestamp(int ts);
    bool signalNextEpoch(int dstRank);

    void HandleEpochCompletion();
    void Evaluation();
    //#########################################
    
    void handlePartitionAllocate(int srcRank);
    void handlePartitionReceive(int srcRank);

    // Coordinator Policy resides here
    virtual PartitionMetadata AllocatePartition(int srcRank);

    bool sendPartition(PartitionMetadata part, int dstRank);
    PartitionMetadata receivePartition(int srcRank);
    void syncPartitionMetadata(PartitionMetadata &part);

    void printCoordinatorState();

public:
    explicit Coordinator(
        std::shared_ptr<c10d::ProcessGroupGloo> pg,
        int num_partitions,
        int num_workers,
        int num_epochs,
        int epochs_per_eval,
        bool gpu) : pg_(pg),
                    num_partitions_(num_partitions),
                    num_workers_(num_workers),
                    tag_generator_(num_workers),
                    num_epochs_(num_epochs),
                    epochs_per_eval_(epochs_per_eval),
                    timestamp_(0),
                    epoch_timer_(gpu)
    {
        // setup
        available_partitions_.clear();
        in_process_partitions_.resize(num_workers_, vector<int>(num_partitions_, 0));
        processed_interactions_.resize(num_partitions_, vector<int>(num_partitions_, 0));
        for (int i = 0; i < num_partitions_; i++)
        {
            available_partitions_.push_back(PartitionMetadata(i, -1, timestamp_, num_partitions_));
        }
        perf_metrics_label_ = "[Performance Metrics]";
    }

    void start_working();
    void stop_working(){}
};
int coordinator_main(int argc, char *argv[]);

#endif