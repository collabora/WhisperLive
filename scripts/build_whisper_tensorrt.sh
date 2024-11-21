#!/bin/bash

download_and_build_model() {
    local model_name="$1"
    local model_url=""

    case "$model_name" in
        "tiny.en")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt"
            ;;
        "tiny")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
            ;;
        "base.en")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt"
            ;;
        "base")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
            ;;
        "small.en")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt"
            ;;
        "small")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
            ;;
        "medium.en")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt"
            ;;
        "medium")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
            ;;
        "large-v1")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt"
            ;;
        "large-v2")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
            ;;
        "large-v3" | "large")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
            ;;
        "large-v3-turbo" | "turbo")
            model_url="https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt"
            ;;
        *)
            echo "Invalid model name: $model_name"
            exit 1
            ;;
    esac

    if [ "$model_name" == "turbo" ]; then
        model_name="large-v3-turbo"
    fi

    local inference_precision="float16"
    local weight_only_precision="${2:-float16}"
    local max_beam_width=4
    local max_batch_size=1

    echo "Downloading $model_name..."
    # wget --directory-prefix=assets "$model_url"
    # echo "Download completed: ${model_name}.pt"
    if [ ! -f "assets/${model_name}.pt" ]; then
        wget --directory-prefix=assets "$model_url"
        echo "Download completed: ${model_name}.pt"
    else
        echo "${model_name}.pt already exists in assets directory."
    fi

    local sanitized_model_name="${model_name//./_}"
    local checkpoint_dir="whisper_${sanitized_model_name}_weights_${weight_only_precision}"
    local output_dir="whisper_${sanitized_model_name}_${weight_only_precision}"
    echo "$output_dir"
    echo "Converting model weights for $model_name..."
    python3 convert_checkpoint.py \
        $( [[ "$weight_only_precision" == "int8" || "$weight_only_precision" == "int4" ]] && echo "--use_weight_only --weight_only_precision $weight_only_precision" ) \
        --output_dir "$checkpoint_dir" --model_name "$model_name"
    
    echo "Building encoder for $model_name..."
    trtllm-build \
        --checkpoint_dir "${checkpoint_dir}/encoder" \
        --output_dir "${output_dir}/encoder" \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_batch_size "$max_batch_size" \
        --gemm_plugin disable \
        --bert_attention_plugin "$inference_precision" \
        --max_input_len 3000 \
        --max_seq_len 3000
    
    echo "Building decoder for $model_name..."
    trtllm-build \
        --checkpoint_dir "${checkpoint_dir}/decoder" \
        --output_dir "${output_dir}/decoder" \
        --moe_plugin disable \
        --enable_xqa disable \
        --max_beam_width "$max_beam_width" \
        --max_batch_size "$max_batch_size" \
        --max_seq_len 200 \
        --max_input_len 14 \
        --max_encoder_input_len 3000 \
        --gemm_plugin "$inference_precision" \
        --bert_attention_plugin "$inference_precision" \
        --gpt_attention_plugin "$inference_precision"

    echo "TensorRT LLM engine built for $model_name."
    echo "========================================="
    echo "Model is located at: $(pwd)/$output_dir"
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path-to-tensorrt-examples-dir> [model-name]"
    exit 1
fi

tensorrt_examples_dir="$1"
model_name="${2:-small.en}"
weight_only_precision="${3:-float16}"  # Default to float16 if not provided

cd $tensorrt_examples_dir/whisper
pip install --no-deps -r requirements.txt

download_and_build_model "$model_name" "$weight_only_precision"
