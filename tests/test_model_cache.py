"""Tests for model cache and hot-swap functionality."""

import unittest
from unittest.mock import patch, MagicMock
from whisper_live.model_cache import ModelCache


class FakeModel:
    """Stub for WhisperModel."""
    def __init__(self, name, **kwargs):
        self.name = name


class TestModelCache(unittest.TestCase):
    def _make_cache(self, max_models=3, default_model="small"):
        cache = ModelCache(max_models=max_models, default_model=default_model)
        # Patch _load_model to avoid actual model loading
        cache._load_model = lambda name, device, ct: FakeModel(name, device=device, compute_type=ct)
        return cache

    def test_default_model(self):
        cache = self._make_cache(default_model="tiny")
        model = cache.get()
        assert model.name == "tiny"

    def test_specific_model(self):
        cache = self._make_cache()
        model = cache.get("large-v3")
        assert model.name == "large-v3"

    def test_cache_hit(self):
        cache = self._make_cache()
        m1 = cache.get("small", "cpu", "int8")
        m2 = cache.get("small", "cpu", "int8")
        assert m1 is m2

    def test_different_devices_different_entries(self):
        cache = self._make_cache()
        m1 = cache.get("small", "cpu", "int8")
        m2 = cache.get("small", "cpu", "float16")
        assert m1 is not m2

    def test_eviction_when_over_capacity(self):
        cache = self._make_cache(max_models=2)
        cache.get("a")
        cache.get("b")
        cache.get("c")  # should evict "a"
        assert len(cache) == 2
        loaded = [m["model"] for m in cache.list_loaded()]
        assert "a" not in loaded
        assert "b" in loaded
        assert "c" in loaded

    def test_lru_ordering(self):
        cache = self._make_cache(max_models=2)
        cache.get("a")
        cache.get("b")
        cache.get("a")  # touch "a", making "b" LRU
        cache.get("c")  # should evict "b"
        loaded = [m["model"] for m in cache.list_loaded()]
        assert "b" not in loaded
        assert "a" in loaded

    def test_list_loaded(self):
        cache = self._make_cache()
        cache.get("small", "cpu", "int8")
        cache.get("medium", "cpu", "int8")
        loaded = cache.list_loaded()
        assert len(loaded) == 2
        assert loaded[0]["model"] == "small"
        assert loaded[1]["model"] == "medium"

    def test_evict_by_name(self):
        cache = self._make_cache()
        cache.get("small")
        cache.get("medium")
        cache.evict("small")
        assert len(cache) == 1
        loaded = [m["model"] for m in cache.list_loaded()]
        assert "small" not in loaded

    def test_clear(self):
        cache = self._make_cache()
        cache.get("small")
        cache.get("medium")
        cache.clear()
        assert len(cache) == 0

    def test_preload(self):
        cache = self._make_cache()
        cache.preload(["tiny", "small", "medium"])
        assert len(cache) == 3

    def test_max_models_minimum_one(self):
        cache = ModelCache(max_models=0)
        assert cache.max_models == 1


class TestServerModelHotSwap(unittest.TestCase):
    def test_server_has_model_cache_attr(self):
        from whisper_live.server import TranscriptionServer
        server = TranscriptionServer()
        assert server._model_cache is None

    def test_model_cache_source_integration(self):
        """Verify server source uses _model_cache.get() for REST transcription."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert "self._model_cache.get(" in source
        assert "resolved_model" in source

    def test_models_endpoint_in_source(self):
        """Verify /v1/models endpoint exists."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert '"/v1/models"' in source


if __name__ == "__main__":
    unittest.main()
