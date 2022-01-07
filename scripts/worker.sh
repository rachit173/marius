#!/bin/bash
if [ "$#" -ne 6 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

rank=$1
wsz=$2
mode=$3
dataset=$4
master=$5
iface=$6

echo "Normal Execution on $dataset"
./build/worker ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --communication.master=$master --communication.iface=$iface --path.base_directory=/dev/shm/data1_worker_$rank/