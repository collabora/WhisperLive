"""
MLX Whisper Model Wrapper for Apple Silicon

This module provides a wrapper around mlx-whisper for optimized
transcription on Apple Silicon (M1/M2/M3) Macs using the MLX framework.
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MLXSegment:
    """
    Represents a transcribed segment with timing information.

    Attributes:
        start (float): Start time in seconds
        end (float): End time in seconds
        text (str): Transcribed text
        no_speech_prob (float): Probability of no speech (0-1)
    """
    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0


class WhisperMLX:
    """
    Wrapper around mlx-whisper for transcription on Apple Silicon.

    This class provides a consistent interface compatible with WhisperLive's
    backend system while using MLX for hardware-accelerated inference.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/whisper-small.en-mlx",
        path_or_hf_repo: str = None,
    ):
        """
        Initialize the MLX Whisper model.

        Args:
            model_name (str): Model name or size. Can be a standard size like "small.en",
                            "base", "medium", "large-v3", or a HuggingFace repo ID.
            path_or_hf_repo (str, optional): Explicit path or HuggingFace repo.
                                            Overrides model_name if provided.
        """
        try:
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
        except ImportError:
            raise ImportError(
                "mlx-whisper is not installed. Install it with: pip install mlx-whisper"
            )

        self.model_name = path_or_hf_repo if path_or_hf_repo else model_name

        # Map standard model sizes to MLX community models
        self.model_size_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "turbo": "mlx-community/whisper-large-v3-turbo",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
        }

        # Convert standard size to MLX repo if needed
        if self.model_name in self.model_size_map:
            self.model_path = self.model_size_map[self.model_name]
            logging.info(f"Mapping model size '{self.model_name}' to '{self.model_path}'")
        else:
            self.model_path = self.model_name

        logging.info(f"Loading MLX Whisper model: {self.model_path}")
        logging.info("MLX will use Apple Neural Engine and GPU for acceleration")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        vad_filter: bool = False,
        vad_parameters: Optional[dict] = None,
    ) -> List[MLXSegment]:
        """
        Transcribe audio using MLX Whisper.

        Args:
            audio (np.ndarray): Audio data as numpy array (16kHz)
            language (str, optional): Language code (e.g., "en", "es", "fr")
            task (str): Task type - "transcribe" or "translate"
            initial_prompt (str, optional): Initial prompt for the model
            vad_filter (bool): Whether to use VAD filtering (not used in MLX)
            vad_parameters (dict, optional): VAD parameters (not used in MLX)

        Returns:
            List[MLXSegment]: List of transcribed segments with timing
        """
        try:
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1] range if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / 32768.0

            # Prepare transcription options
            transcribe_opts = {
                "path_or_hf_repo": self.model_path,
                "task": task,
                "verbose": False,
            }

            if language:
                transcribe_opts["language"] = language

            if initial_prompt:
                transcribe_opts["initial_prompt"] = initial_prompt

            # Run transcription
            result = self.mlx_whisper.transcribe(
                audio,
                **transcribe_opts
            )

            # Convert result to MLXSegment objects
            segments = []

            if isinstance(result, dict) and "segments" in result:
                for seg in result["segments"]:
                    segment = MLXSegment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", "").strip(),
                        no_speech_prob=seg.get("no_speech_prob", 0.0)
                    )
                    segments.append(segment)
            elif isinstance(result, dict) and "text" in result:
                # If only text is returned (no segments), create a single segment
                segment = MLXSegment(
                    start=0.0,
                    end=len(audio) / 16000.0,  # Calculate duration
                    text=result["text"].strip(),
                    no_speech_prob=0.0
                )
                segments.append(segment)

            return segments

        except Exception as e:
            logging.error(f"MLX transcription failed: {e}")
            raise

    def detect_language(self, audio: np.ndarray) -> tuple:
        """
        Detect the language of the audio.

        Args:
            audio (np.ndarray): Audio data

        Returns:
            tuple: (language_code, probability)
        """
        try:
            # MLX whisper doesn't have a separate language detection API
            # We'll transcribe a small portion and infer from the result
            result = self.mlx_whisper.transcribe(
                audio[:16000],  # First second
                path_or_hf_repo=self.model_path,
                task="transcribe",
                verbose=False,
            )

            # Try to extract language from result
            if isinstance(result, dict):
                lang = result.get("language", "en")
                return (lang, 1.0)

            return ("en", 1.0)

        except Exception as e:
            logging.error(f"Language detection failed: {e}")
            return ("en", 1.0)
