#!/bin/bash

apt-get update && apt-get -y install git git-lfs
git clone -b cuda12.2 https://github.com/makaveli10/TensorRT-LLM.git
cd TensorRT-LLM
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
bash install_mpi4py.sh
source $ENV
