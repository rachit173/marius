#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

rank=$1
wsz=$2
mode=$3
dataset=$4

echo "Normal Execution on $dataset"
marius_worker ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --path.base_directory=data1_worker_$rank/
