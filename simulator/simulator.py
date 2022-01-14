import imp
import numpy as np
import random
import argparse
from coordinator import Coordinator
from worker import Worker

# Event Types
REQUEST_PARTITION = 0       #  --> {R: -1} : Indicates that worker is asking for partition from coordinator
DISPATCH_PARTITION = 1      #  --> {D: Partition #} : Indicates that worker is releasing the partition with given # to coordinator
INTERACTION_COMPLETE = 2    #  --> {I: (i, j)} : Indicates that edge bucket (i, j) has been processed by worker

######################## Simulation code ##################################
# 1. Pick earliest event from one of the worker's event queue based on lowest time
# 2. Process the event --> 
#   a. Update coordinator and worker state
#   b. Insert events that can be triggered by the current event

class Simulator:

    def __init__(self, workers, coordinator):
        self.workers = workers
        self.coordinator = coordinator

        #Execution Stats
        self.finish_time, self.num_events, self.disk_transfers, self.nw_transfers = 0, 0, 0, 0

    def pick_earliest_event(self):
        workers = self.workers
        earliest_event_times = np.array([sorted(worker.events.keys())[0] for worker in workers])
        worker_id = np.argmin(earliest_event_times)
        key = earliest_event_times[worker_id]
        event = workers[worker_id].events[key]
        del workers[worker_id].events[key]    
        return worker_id, key, event

    def process_event(self, timestamp, event, worker_id):
        workers = self.workers
        coordinator = self.coordinator
        disk_transfers, nw_transfers = 0, 0

        request_type = list(event.keys())[0]
        worker = workers[worker_id]
        num_partitions = worker.num_partitions

        if(request_type == REQUEST_PARTITION):
            partition_id, owner = coordinator.allocate_partition(worker_id, coordinator.policy)

            # Update disk and network transfers and partition owner
            if(owner == -1):
                disk_transfers += 1
            elif (owner != worker_id):
                nw_transfers += 1
            else: # partition is present on same node
                pass
            
            # update partition location to worker and put to buffer
            worker.buffer += [partition_id]
            coordinator.partition_holder[partition_id] = worker_id
            
            ###### Add future interaction completion events considering new partition in future timestamps #######
            possible_interactions = []
            for i in worker.buffer:
                if(worker.interactions[i][partition_id] == 0):
                    possible_interactions += [(i, partition_id)]
                if(worker.interactions[partition_id][i] == 0):
                    possible_interactions += [(partition_id, i)]
            if(worker.interactions[partition_id][partition_id] == 0):
                possible_interactions += [(partition_id, partition_id)]
            
            possible_interactions = list(set(possible_interactions))
            possible_interactions = worker.order_interactions(possible_interactions)
            
            worker.add_computation_events(timestamp, possible_interactions)
            ##########################################################################

            #Check if some partition can be evicted immediately as it is not useful
            evict_partition = worker.evict()
            if evict_partition != -1:
                worker.events[timestamp] = {DISPATCH_PARTITION: evict_partition}


        elif (request_type == DISPATCH_PARTITION):
            partition_dispatched = event[request_type]
            
            # update coordinator interactions
            for i in range(num_partitions):
                coordinator.interactions[i][partition_dispatched] = coordinator.interactions[i][partition_dispatched] or worker.interactions[i][partition_dispatched]
                coordinator.interactions[partition_dispatched][i] = coordinator.interactions[partition_dispatched][i] or worker.interactions[partition_dispatched][i]
            
            #Update coordinator partition view
            coordinator.in_process_partitions[worker_id].remove(partition_dispatched)
            coordinator.available_partitions += [partition_dispatched]

            #Update worker buffer
            worker.buffer.remove(partition_dispatched)

            # Put request for partition to coordinator
            worker.handle_eviction(timestamp)

        
        elif (request_type == INTERACTION_COMPLETE):
            interaction = event[request_type]
            
            #Update worker state
            worker.interactions[interaction[0]][interaction[1]] = 1

            #Check if some partition can be evicted
            evict_partition = worker.evict()
            if evict_partition != -1:
                worker.events[timestamp] = {DISPATCH_PARTITION: evict_partition}

        ### Blanket event addition if buffer not full
        if( len(worker.buffer) < worker.buffer_capacity):
            worker.events[timestamp] = {REQUEST_PARTITION: -1}

        return disk_transfers, nw_transfers

    def print_all_events(self):
        for i in range(len(self.workers)):
            print("Worker {}:".format(i))
            self.workers[i].print_events()

    #Simulation loop
    def simulate(self):
        while(not self.coordinator.can_terminate()): # Put termination condition
            worker_id, timestamp, event = self.pick_earliest_event()
            self.finish_time = timestamp
            new_disk_transfers, new_nw_transfers = self.process_event(timestamp, event, worker_id)
            
            # Track stats
            self.num_events += 1
            self.disk_transfers += new_disk_transfers
            self.nw_transfers += new_nw_transfers
        print("Finish time: {}".format(self.finish_time))
        print("Number of 'events': {}".format(self.num_events))
        print("Number of disk transfers: {}".format(self.disk_transfers))
        print("Number of network transfers: {}".format(self.nw_transfers))

############### Input parsing and initialization #####################
parser = argparse.ArgumentParser()
parser.add_argument('--num_partitions', type=int, required=True)
parser.add_argument('--buffer_capacity', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--policy', type=str, required=True)

parser.add_argument('--computation_mean', type=int, required=True)
parser.add_argument('--computation_std', type=int, required=True)

args = parser.parse_args()
num_partitions =  args.num_partitions
buffer_capacity = args.buffer_capacity
num_workers = args.num_workers
policy = args.policy

#TODO: Include disk and nw transfer times if reqd
# Model Computation time per edge block
time_per_edge_block_mean, time_per_edge_block_std = args.computation_mean, args.computation_std

workers = [Worker(num_partitions, buffer_capacity, time_per_edge_block_mean, time_per_edge_block_std, -1, -1) for i in range(num_workers)]
coordinator = Coordinator(num_partitions, num_workers, policy)

# Put initial event in all worker buffers for requesting partition
for i in range(num_workers):
    request_time = np.random.uniform(0, 0.01)
    workers[i].events[request_time] = {REQUEST_PARTITION: -1}
    workers[i].earliest_available_time = request_time

simulator = Simulator(workers, coordinator)
simulator.simulate()