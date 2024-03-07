#!/bin/bash -e

apt-get update && apt-get -y install git git-lfs
git clone --depth=1 -b cuda12.2 https://github.com/makaveli10/TensorRT-LLM.git
cd TensorRT-LLM
git checkout main
git submodule update --init --recursive
git lfs install
git lfs pull

# do not reinstall CUDA (our base image provides the same exact versions)
patch -p1 <<EOF
diff --git a/docker/common/install_tensorrt.sh b/docker/common/install_tensorrt.sh
index 2dcb0a6..3a27e03 100644
--- a/docker/common/install_tensorrt.sh
+++ b/docker/common/install_tensorrt.sh
@@ -35,19 +35,7 @@ install_ubuntu_requirements() {
     dpkg -i cuda-keyring_1.0-1_all.deb

     apt-get update
-    if [[ $(apt list --installed | grep libcudnn8) ]]; then
-      apt-get remove --purge -y libcudnn8*
-    fi
-    if [[ $(apt list --installed | grep libnccl) ]]; then
-      apt-get remove --purge -y --allow-change-held-packages libnccl*
-    fi
-    if [[ $(apt list --installed | grep libcublas) ]]; then
-      apt-get remove --purge -y --allow-change-held-packages libcublas*
-    fi
-    CUBLAS_CUDA_VERSION=$(echo $CUDA_VER | sed 's/\./-/g')
     apt-get install -y --no-install-recommends libcudnn8=${CUDNN_VER} libcudnn8-dev=${CUDNN_VER}
-    apt-get install -y --no-install-recommends libnccl2=${NCCL_VER} libnccl-dev=${NCCL_VER}
-    apt-get install -y --no-install-recommends libcublas-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER} libcublas-dev-${CUBLAS_CUDA_VERSION}=${CUBLAS_VER}
     apt-get clean
     rm -rf /var/lib/apt/lists/*
 }
EOF

cd docker/common/
export BASH_ENV=${BASH_ENV:-/etc/bash.bashrc}
export ENV=${ENV:-/etc/shinit_v2}
bash install_base.sh
bash install_cmake.sh
source $ENV
bash install_ccache.sh
# later on TensorRT-LLM will force reinstall this version anyways
pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.1.0
bash install_tensorrt.sh
bash install_polygraphy.sh
source $ENV

cd /root/TensorRT-LLM/docker/common/
bash install_mpi4py.sh
source $ENV
