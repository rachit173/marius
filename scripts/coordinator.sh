#!/bin/bash
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# cd build/ &&  cmake --build . --target worker coordinator -- -j 5 && cd ..
# BASE_DIR="/mnt/data/Work/marius"
# rm -f $BASE_DIR/rendezvous_checkpoint
# rm -f rendezvous_checkpoint
rank=$1
wsz=$2
mode=$3
dataset=$4
master=$5

echo "Normal Execution on $dataset"
marius_coordinator ./examples/training/configs/$dataset.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC --communication.master=$master