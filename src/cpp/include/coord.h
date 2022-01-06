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
public:
    void start_working();
    void stop_working();
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
}
#endif