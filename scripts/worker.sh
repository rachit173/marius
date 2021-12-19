#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

rank=$1
wsz=$2
mode=$3
dataset=$4
if [[ "$mode" = "gdb" ]]; then
    echo "GDB Execution on $dataset"
    gdb --args ./build/worker ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --path.base_directory=data1_worker_$rank/
else
    echo "Normal Execution on $dataset"
    ./build/worker ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --path.base_directory=data1_worker_$rank/
fi