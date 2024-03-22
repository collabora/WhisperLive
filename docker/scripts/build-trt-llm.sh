#!/bin/bash -e

export ENV=${ENV:-/etc/shinit_v2}
source $ENV

CUDA_ARCH="${CUDA_ARCH:-89-real;90-real}"

cd /root/TensorRT-LLM
python3 scripts/build_wheel.py --clean --cuda_architectures "$CUDA_ARCH" --trt_root /usr/local/tensorrt
