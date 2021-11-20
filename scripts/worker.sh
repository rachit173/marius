rank=$1
ws=$2
./build/worker ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$ws --communication.prefix=ABC
