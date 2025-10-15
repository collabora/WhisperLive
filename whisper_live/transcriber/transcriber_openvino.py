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
            initial_prompt=None,
            cpu_threads=0,
            cache_path=None
        ):
        """
        Initialize WhisperOpenVINO transcriber.

        Args:
            model_id (str): HuggingFace model ID for OpenVINO model. Defaults to "OpenVINO/whisper-tiny-int8-ov".
            device (str): Target device (CPU, GPU, etc.). Defaults to "CPU".
            language (str): Language code for transcription. Defaults to "en".
            task (str): Task type ("transcribe" or "translate"). Defaults to "transcribe".
            initial_prompt (str, optional): Initial prompt for transcription. Defaults to None.
            cpu_threads (int): Number of CPU threads for inference. 0 means auto-detect. Defaults to 0.
            cache_path (str, optional): Path for OpenVINO compilation cache directory. Defaults to None.
        """
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
        self.initial_prompt = initial_prompt

    def _setup_cache(self, device, cache_path):
        """Setup compilation cache if specified."""
        if cache_path is None:
            return

        if device != "CPU":
            logging.warning(
                f"[OpenVINO] Cache not supported for device '{device}' with WhisperPipeline")
            return

        cache_dir = Path(cache_path) / "openvino_compiled"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["OV_CPU_ENABLE_MODEL_CACHE"] = "1"
        os.environ["OPENVINO_CACHE_DIR"] = str(cache_dir)
        logging.info(
            f"[OpenVINO] CPU cache enabled via environment variables: {cache_dir}")

    def _get_cpu_config(self, cpu_threads):
        """Get CPU-specific optimizations."""
        return {
            "INFERENCE_NUM_THREADS": cpu_threads,
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": 1,
            "SCHEDULING_CORE_TYPE": "ANY_CORE",
            "ENABLE_HYPER_THREADING": True,
            "ENABLE_CPU_PINNING": True,
            "CPU_DENORMALS_OPTIMIZATION": True,
            "INFERENCE_PRECISION_HINT": "f16",
        }

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
        self._setup_cache(device, cache_path)

        if device != "CPU":
            return {}

        return self._get_cpu_config(cpu_threads)

    def transcribe(self, input_audio):
        # Build generate parameters
        generate_kwargs = {
            "return_timestamps": True,
            "task": self.task
        }

        # Only add language if specified (None = auto-detect)
        if self.language is not None:
            generate_kwargs["language"] = self.language

        # Only add initial_prompt if specified
        if self.initial_prompt is not None:
            generate_kwargs["initial_prompt"] = self.initial_prompt

        outputs = self.model.generate(input_audio, **generate_kwargs)
        outputs = [seg for seg in outputs.chunks]
        return outputs
