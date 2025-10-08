import librosa
import os

import openvino_genai as ov_genai
import huggingface_hub as hf_hub


class WhisperOpenVINO(object):
    def __init__(self, model_id="OpenVINO/whisper-tiny-int8-ov", device="CPU", language="en", task="transcribe", cpu_threads=None):
        # Use HuggingFace cache exclusively - snapshot_download handles everything
        model_path = hf_hub.snapshot_download(model_id)

        # Create pipeline with optimized CPU configuration if specified
        if cpu_threads is not None and device == "CPU":
            # Optimized configuration for real-time CPU inference
            # Based on best practices for low-latency speech recognition
            self.model = ov_genai.WhisperPipeline(
                str(model_path),
                device=device,
                INFERENCE_NUM_THREADS=cpu_threads, # Number of CPU threads
                PERFORMANCE_HINT="LATENCY",        # Optimize for low latency LATENCY, THROUGHPUT
                NUM_STREAMS=1,                     # Single stream for real-time
                SCHEDULING_CORE_TYPE="ANY_CORE",   # Use P-cores only (hybrid CPUs) ANY_CORE, PCORE_ONLY, ECORE_ONLY
                ENABLE_HYPER_THREADING=True,       # Désactiver HT = moins de contention
                ENABLE_CPU_PINNING=True,           # Pin threads to cores (Linux)
                CPU_DENORMALS_OPTIMIZATION=True,   # Optimize denormal numbers
                INFERENCE_PRECISION_HINT="f16",    # FP16 interne (plus rapide que FP32) f16/f32/bf16
            )
        else:
            self.model = ov_genai.WhisperPipeline(
                str(model_path),
                device=device
            )

        self.language = language
        self.task = task

    def transcribe(self, input_audio):
        outputs = self.model.generate(input_audio, return_timestamps=True, language=self.language, task=self.task)
        outputs = [seg for seg in outputs.chunks]
        return outputs
