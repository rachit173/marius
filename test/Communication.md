In the build directory, 

For running workers,

./worker ../examples/training/configs/fb15k_gpu.ini --communication.rank=0 --communication.world_size=2


./worker ../examples/training/configs/fb15k_gpu.ini --communication.rank=1 --communication.world_size=3

For running coordinator,
./coordinator ../examples/training/configs/fb15k_gpu.ini --communication.rank=2 --communication.world_size=3


If there are n workers, then world size is n+1 and the coordinator rank is set to n.