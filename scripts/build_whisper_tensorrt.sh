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
        *)
            echo "Invalid model name: $model_name"
            exit 1
            ;;
    esac

    echo "Downloading $model_name..."
    # wget --directory-prefix=assets "$model_url"
    # echo "Download completed: ${model_name}.pt"
    if [ ! -f "assets/${model_name}.pt" ]; then
        wget --directory-prefix=assets "$model_url"
        echo "Download completed: ${model_name}.pt"
    else
        echo "${model_name}.pt already exists in assets directory."
    fi

    local output_dir="whisper_${model_name//./_}"
    echo "$output_dir"
    echo "Running build script for $model_name with output directory $output_dir"
    python3 build.py --output_dir "$output_dir" --use_gpt_attention_plugin --use_gemm_plugin  --use_bert_attention_plugin --enable_context_fmha --model_name "$model_name"
    echo "Whisper $model_name TensorRT engine built."
    echo "========================================="
    echo "Model is located at: $(pwd)/$output_dir"
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path-to-tensorrt-examples-dir> [model-name]"
    exit 1
fi

tensorrt_examples_dir="$1"
model_name="${2:-small.en}"

cd $1/whisper
pip install --no-deps -r requirements.txt

download_and_build_model "$model_name"
