cd build/ &&  cmake --build . --target worker coordinator -- -j 5 && cd ..
rm rendezvous_checkpoint
rank=$1
wsz=$2
./build/coordinator ./examples/training/configs/fb15k_cpu.ini --communication.rank=$rank --communication.world_size=$wsz --communication.prefix=ABC
