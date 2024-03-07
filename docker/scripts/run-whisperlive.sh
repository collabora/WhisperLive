#!/bin/bash -e
echo "MODEL_SIZE is set to: $MODEL_SIZE"

test -f /etc/shinit_v2 && source /etc/shinit_v2

echo "Running build-models.sh..."
cd /root/scratch-space/
./build-whisper-tensorrt.sh /root/TensorRT-LLM-examples/ $MODEL_SIZE

whisper_model_trt="whisper_${MODEL_SIZE//./_}"

echo "$whisper_model_trt"

cd /root/WhisperLive
if [[ $MODEL_SIZE == *".en" ]]; then
    # If MODEL_SIZE ends with '.en', run without -m option
    exec python3 run_server.py -p 9090 -b tensorrt \
        -trt /root/scratch-space/models/"$whisper_model_trt"
else
    # If MODEL_SIZE does not end with '.en', run with -m option
    exec python3 run_server.py -p 9090 -b tensorrt \
        -trt /root/scratch-space/models/"$whisper_model_trt" \
        -m
fi
