#!/bin/bash

apt-get update && apt-get -y install git git-lfs
git clone https://github.com/makaveli10/TensorRT-LLM.git
cd TensorRT-LLM
git checkout main
git submodule update --init --recursive
git lfs install
git lfs pull

cd docker/common/
export BASH_ENV=${BASH_ENV:-/etc/bash.bashrc}
export ENV=${ENV:-/etc/shinit_v2}
bash install_base.sh
bash install_cmake.sh
bash install_ccache.sh
bash install_tensorrt.sh
bash install_polygraphy.sh
source $ENV

cd /root
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.5.tar.gz
tar -xzvf openmpi-3.1.5.tar.gz
rm -rf openmpi-3.1.5.tar.gz
cd openmpi-3.1.5
./configure --prefix=/usr/local/
sudo make all install
echo 'export PATH=$PATH:/usr/local/bin' >> "${ENV}"
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> "${ENV}"
source $ENV

cd /home/TensorRT-LLM/docker/common/
bash install_mpi4py.sh
source $ENV

cuda_architectures="${1:-89-real}"


cd /home/TensorRT-LLM
python3 scripts/build_wheel.py --cuda_architectures "${cuda_architectures}" --clean --trt_root /usr/local/tensorrt
pip install build/tensorrt_llm-*.whl
pip install --no-deps openai-whisper