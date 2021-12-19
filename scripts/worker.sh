#!/bin/bash
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

rank=$1
wsz=$2
mode=$3
dataset=$4
master=$5

echo "Normal Execution on $dataset"
marius_worker ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --communication.master=$master --path.base_directory=data1_worker_$rank/