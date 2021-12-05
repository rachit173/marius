#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

cd build/ &&  cmake --build . --target worker coordinator -- -j 5 && cd ..
BASE_DIR="/proj/uwmadison744-f21-PG0/groups/g007"
rm -f $BASE_DIR/rendezvous_checkpoint
rm -f rendezvous_checkpoint
rank=$1
wsz=$2
mode=$3
dataset=$4
if [[ "$mode" = "gdb" ]]; then
    echo "GDB Execution on $dataset"
    gdb --args ./build/coordinator ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
else
    echo "Normal Execution on $dataset"
    ./build/coordinator ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
fi