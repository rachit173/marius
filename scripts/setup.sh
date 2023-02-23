#!/bin/bash
BASE_DIR=/mnt/data/Work/
mkdir -p $BASE_DIR
cd $BASE_DIR

sudo apt update
sudo apt install -y g++ make wget unzip git python3-pip

# install gcc-8
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# install cmake 3.20
wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.sh
sudo mkdir /opt/cmake
sudo sh cmake-3.20.0-linux-x86_64.sh --skip-license --prefix=/opt/cmake/
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake


# install pytorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.10.0%2Bcpu.zip
unzip libtorch-shared-with-deps-1.10.0+cpu.zip
rm libtorch-shared-with-deps-1.10.0+cpu.zip

# install gloo
git clone https://github.com/facebookincubator/gloo.git
cd gloo
mkdir -p build
cd build
cmake ..
make
cp gloo/config.h ../gloo/

# include gloo headers
ln -s $BASE_DIR/gloo/gloo/ $BASE_DIR/libtorch/include/gloo

# build marius
cd $BASE_DIR
# git clone https://github.com/rachit173/marius.git
cd marius
rm -rf build
mkdir build
cd build
cmake -DMARIUS_TORCH_DIR=$BASE_DIR/libtorch -DMARUIS_GLOO_DIR=$BASE_DIR/gloo ..
make -j5

# build python wrappers
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade pillow
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

ln -s $BASE_DIR/gloo/gloo/ ~/.local/lib/python3.6/site-packages/torch/include/gloo

cd $BASE_DIR/marius
python3 -m pip install .

