"""
Model cache for hot-swapping Whisper models without server restart.

Maintains an LRU cache of loaded WhisperModel instances, allowing
per-request model selection without re-loading on every call.
"""

import logging
import threading
from collections import OrderedDict
from typing import Optional


class ModelCache:
    """Thread-safe LRU cache for WhisperModel instances.

    Args:
        max_models: Maximum number of models to keep loaded simultaneously.
            When exceeded, the least recently used model is evicted.
            Default 3.
        default_model: Default model name/path to use when none is specified.
    """

    def __init__(self, max_models: int = 3, default_model: str = "small"):
        self.max_models = max(1, max_models)
        self.default_model = default_model
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model_name: Optional[str] = None, device: str = "cpu",
            compute_type: str = "int8"):
        """Get or load a WhisperModel, returning it from cache if available.

        Args:
            model_name: Model name or path. Uses default_model if None.
            device: Device to load on ("cpu" or "cuda").
            compute_type: Compute type ("int8", "float16", etc.).

        Returns:
            A loaded WhisperModel instance.
        """
        name = model_name or self.default_model
        cache_key = (name, device, compute_type)

        with self._lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                logging.info(f"Model cache hit: {name} ({device}/{compute_type})")
                return self._cache[cache_key]

        # Load outside the lock to avoid blocking other requests
        model = self._load_model(name, device, compute_type)

        with self._lock:
            # Double-check in case another thread loaded it
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

            self._cache[cache_key] = model
            self._cache.move_to_end(cache_key)

            # Evict LRU if over capacity
            while len(self._cache) > self.max_models:
                evicted_key, _ = self._cache.popitem(last=False)
                logging.info(f"Model cache eviction: {evicted_key[0]}")

        return model

    def _load_model(self, model_name, device, compute_type):
        """Load a WhisperModel instance."""
        from faster_whisper import WhisperModel
        logging.info(f"Loading model: {model_name} ({device}/{compute_type})")
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    def preload(self, model_names, device="cpu", compute_type="int8"):
        """Preload multiple models into cache.

        Args:
            model_names: List of model names to preload.
            device: Device to load on.
            compute_type: Compute type.
        """
        for name in model_names:
            self.get(name, device, compute_type)

    def list_loaded(self):
        """Return list of currently loaded model keys."""
        with self._lock:
            return [
                {"model": k[0], "device": k[1], "compute_type": k[2]}
                for k in self._cache.keys()
            ]

    def evict(self, model_name: str):
        """Evict a specific model from cache."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[0] == model_name]
            for key in keys_to_remove:
                del self._cache[key]

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()

    def __len__(self):
        with self._lock:
            return len(self._cache)
