#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

cd build/ &&  cmake --build . --target worker coordinator -- -j 5 && cd ..
rm rendezvous_checkpoint
rank=$1
wsz=$2
mode=$3
if [[ "$mode" = "gdb" ]]; then
    echo "GDB Execution"
    gdb --args ./build/coordinator ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
else
    echo "Normal Execution"
    ./build/coordinator ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
fi