#!/bin/bash
cd ~/Work/marius/build
rm -rf CMakeCache.txt
cmake -DMARIUS_TORCH_DIR=~/Work/libtorch -DMARUIS_GLOO_DIR=~/Work/gloo ..
make -j4
