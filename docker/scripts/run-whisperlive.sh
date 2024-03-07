#!/bin/bash -e

echo "MODEL_SIZE is set to: $MODEL_SIZE"

test -f /etc/shinit_v2 && source /etc/shinit_v2

echo "Running build-models.sh..."
cd /root/scratch-space/
./build-whisper-tensorrt.sh /root/TensorRT-LLM-examples/ small.en

cd /root/WhisperLive
exec python3 run_server.py -p 9090 -b tensorrt \
                -trt /root/scratch-space/models/whisper_small_en \
