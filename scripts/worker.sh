#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

rank=$1
wsz=$2
mode=$3
if [[ "$mode" = "gdb" ]]; then
    echo "GDB Execution"
    gdb --args ./build/worker ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
else
    echo "Normal Execution"
    ./build/worker ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
fi