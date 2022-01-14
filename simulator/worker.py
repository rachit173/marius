import numpy as np
import random

# Event Types
REQUEST_PARTITION = 0       #  --> {R: -1} : Indicates that worker is asking for partition from coordinator
DISPATCH_PARTITION = 1      #  --> {D: Partition #} : Indicates that worker is releasing the partition with given # to coordinator
INTERACTION_COMPLETE = 2    #  --> {I: (i, j)} : Indicates that edge bucket (i, j) has been processed by worker

class Worker:
    def __init__(self, num_partitions, buffer_capacity, edge_block_time, edge_block_time_std, disk_transfer_time, nw_transfer_time, num_epochs = 1):
        #Parameters
        self.num_partitions = num_partitions
        self.buffer_capacity = buffer_capacity
        
        # Times
        self.edge_block_time = edge_block_time
        self.edge_block_time_std = edge_block_time_std
        self.disk_transfer_time = disk_transfer_time
        self.nw_transfer_time = nw_transfer_time
        
        self.num_epochs = num_epochs
        
        # Data structures
        self.events = {} # Event of type: REQUEST_PARTITION, DISPATCH_PARTITION, INTERACTION_COMPLETE
        self.interactions = np.zeros((num_partitions, num_partitions))
        self.buffer = []
        self.earliest_available_time = 0 # Time at which the next computation starts on worker
    
    # Prints current state of events yet to be processed for this worker
    def print_events(self):
        for key in sorted(self.events.keys()):
            print( "{} : {}".format(key, self.events[key]) )

    # Given list of possible interactions, add them into the event queue with a completion timestamp in future
    def add_computation_events(self, timestamp, possible_interactions):
        event_time = self.earliest_available_time
        for interaction in possible_interactions:
            
            # Randomness in computation time
            computation_time = np.random.normal(self.edge_block_time, self.edge_block_time_std)

            event_time += computation_time

            self.events[event_time] = {INTERACTION_COMPLETE: interaction}
        self.earliest_available_time = event_time

    def handle_eviction(self, timestamp):
        # Ideally should be true all the time
        if(len(self.buffer) < self.buffer_capacity):
            self.events[timestamp] = {REQUEST_PARTITION: -1}

    ######################## Insert Worker policy here ###########################
    def evict(self):
        if(len(self.buffer) < self.buffer_capacity):
            return -1
        
        # Choose randomly from all eviction candidates
        evict_candidates = []
        for partition_id in self.buffer:
            possible = True
            for partition_id2 in self.buffer:
                if self.interactions[partition_id][partition_id2] == 0 or self.interactions[partition_id2][partition_id] == 0:
                    possible = False
                    break
            if possible:
                evict_candidates += [partition_id]
        if evict_candidates == []:
            return -1
        evict_partition = evict_candidates[random.randint(0, len(evict_candidates) - 1)]
        print("Mark for eviction -> partition {}".format(evict_partition))
        return evict_partition

    def order_interactions(self, edge_blocks):
        return edge_blocks