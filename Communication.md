For development purpose we can use examples/training/configs/fb15k_gpu.ini,

For preparing dataset. 
1. mkdir <marius_root_dir>/output_dir/
2. cd output_dir && curl -O https://dl.fbaipublicfiles.com/starspace/fb15k.tgz
3. cd <marius_root_dir> && marius_preprocess output_dir/ --dataset fb15k

In the build directory, 

For running workers,

./worker ../examples/training/configs/fb15k_gpu.ini --communication.rank=0 --communication.world_size=2


./worker ../examples/training/configs/fb15k_gpu.ini --communication.rank=1 --communication.world_size=3

For running coordinator,
./coordinator ../examples/training/configs/fb15k_gpu.ini --communication.rank=2 --communication.world_size=3


If there are n workers, then world size is n+1 and the coordinator rank is set to n.