import librosa
import os
import logging
from pathlib import Path

import openvino_genai as ov_genai
import huggingface_hub as hf_hub


class WhisperOpenVINO(object):
    def __init__(
            self,
            model_id="OpenVINO/whisper-tiny-int8-ov",
            device="CPU",
            language="en",
            task="transcribe",
            cpu_threads=0,
            cache_path=None
        ):
        # Use HuggingFace cache exclusively - snapshot_download handles everything
        model_path = hf_hub.snapshot_download(model_id)

        # Build OpenVINO configuration
        config = self._build_config(device, cpu_threads, cache_path)

        # Create pipeline with unified configuration
        self.model = ov_genai.WhisperPipeline(
            str(model_path),
            device=device,
            **config
        )

        self.language = language
        self.task = task

    def _build_config(self, device, cpu_threads, cache_path):
        """
        Build OpenVINO configuration based on device and parameters.

        Args:
            device: Target device (CPU, GPU, etc.)
            cpu_threads: Number of CPU threads (0=auto)
            cache_path: Path for model cache

        Returns:
            dict: OpenVINO configuration dictionary
        """
        config = {}

        # Setup compilation cache if specified
        if cache_path is not None:
            cache_dir = Path(cache_path) / "openvino_compiled"
            cache_dir.mkdir(parents=True, exist_ok=True)
            config["CACHE_DIR"] = str(cache_dir)
            logging.info(f"[OpenVINO] Model cache enabled: {cache_dir}")

        if device != "CPU":
            return config

        # Add CPU-specific optimizations
        # Optimized configuration for real-time CPU inference
        # Based on OpenVINO 2025 best practices for low-latency speech recognition
        config.update({
            # Threading configuration
            "INFERENCE_NUM_THREADS": cpu_threads,  # Number of CPU threads (default: 0=auto)
            # Performance optimization
            "PERFORMANCE_HINT": "LATENCY",         # Performance mode
                                                   # Options: LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT
            "NUM_STREAMS": 1,                      # Number of parallel inference streams
                                                   # Options: 1 (real-time), >1 (batch), AUTO
            # CPU scheduling (hybrid Intel CPUs: P-cores/E-cores)
            "SCHEDULING_CORE_TYPE": "ANY_CORE",    # Core type selection
                                                   # Options: ANY_CORE, PCORE_ONLY, ECORE_ONLY
            "ENABLE_HYPER_THREADING": True,        # Use logical cores (SMT/HT)
                                                   # Options: True, False
            "ENABLE_CPU_PINNING": True,            # Pin threads to physical cores (Linux only)
                                                   # Options: True, False
            # Intel CPU-specific optimizations
            "CPU_DENORMALS_OPTIMIZATION": True,    # Treat denormal floats as zero (faster, less accurate)
                                                   # Options: True, False
            # Precision hint (internal computation)
            "INFERENCE_PRECISION_HINT": "f16",     # Internal precision for faster computation
                                                   # Options: f32 (accurate), f16 (fast), bf16 (balanced)
        })

        return config

    def transcribe(self, input_audio):
        outputs = self.model.generate(input_audio, return_timestamps=True, language=self.language, task=self.task)
        outputs = [seg for seg in outputs.chunks]
        return outputs
