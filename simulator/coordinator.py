import numpy as np
import random

class Coordinator:
    def __init__(self, num_partitions, num_workers, policy, num_epochs = 1):
        self.num_partitions = num_partitions
        self.num_epochs = num_epochs
        self.policy = policy

        self.interactions = np.zeros((num_partitions, num_partitions))
        self.available_partitions = [i for i in range(num_partitions)]
        self.in_process_partitions = [[] for i in range(num_workers)]
        self.partition_holder = {} # Locations of available partitions
        for i in range(num_partitions):
            self.partition_holder[i] = -1  # -1 --> partition is present on disk
    
    def random_allocation(self):
        length = len(self.available_partitions)
        # Return random partition
        idx = random.randint(0, length - 1)
        return self.available_partitions[idx]
    
    def max_work_possible(self, worker_id):
        partition_score = np.zeros(len(self.available_partitions))
        for idx in range(len(self.available_partitions)):
            partition = self.available_partitions[idx]
            if self.interactions[partition][partition] == 0:
                partition_score[idx] += 1
            for p in self.in_process_partitions[worker_id]:
                if self.interactions[p][partition] == 0:
                    partition_score[idx] += 2
        
        # Get partition with the max score
        reqd_idx = np.argmax(partition_score)
        partition_id = self.available_partitions[reqd_idx]
        return partition_id

    ######################## Insert Coordinator allocation policy here ###########################
    # def my_policy(self, worker_id):
    ##############################################################################################

    def allocate_partition(self, worker_id, policy = "random"):
        if(len(self.available_partitions) == 0):
            return -1
        if policy == "random":
            partition_id = self.random_allocation()
        elif policy == "max_work_possible":
            partition_id = self.max_work_possible(worker_id)
        ### Put case for your policy

        print("Allocated partition: {}".format(partition_id))
        owner = self.partition_holder[partition_id]

        self.available_partitions.remove(partition_id)
        self.in_process_partitions[worker_id] += [partition_id]

        return partition_id, owner

    def can_terminate(self):
        total_done = np.sum(self.interactions)
        return (total_done == self.num_partitions * self.num_partitions)