"""Tests for plugin architecture."""

import unittest
from whisper_live.plugins import PluginRegistry, PluginEntry, default_registry


class TestPluginEntry(unittest.TestCase):
    def test_creation(self):
        entry = PluginEntry("test", lambda s: s, priority=10, enabled=True)
        assert entry.name == "test"
        assert entry.priority == 10
        assert entry.enabled is True


class TestPluginRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = PluginRegistry()

    def test_add_and_contains(self):
        self.registry.add("test", lambda s: s)
        assert "test" in self.registry
        assert len(self.registry) == 1

    def test_duplicate_name_raises(self):
        self.registry.add("test", lambda s: s)
        with self.assertRaises(ValueError):
            self.registry.add("test", lambda s: s)

    def test_remove(self):
        self.registry.add("test", lambda s: s)
        self.registry.remove("test")
        assert "test" not in self.registry

    def test_remove_nonexistent_no_error(self):
        self.registry.remove("nonexistent")

    def test_enable_disable(self):
        self.registry.add("test", lambda s: s, enabled=False)
        assert not self.registry._plugins["test"].enabled
        self.registry.enable("test")
        assert self.registry._plugins["test"].enabled
        self.registry.disable("test")
        assert not self.registry._plugins["test"].enabled

    def test_list_plugins_sorted_by_priority(self):
        self.registry.add("b", lambda s: s, priority=20)
        self.registry.add("a", lambda s: s, priority=10)
        self.registry.add("c", lambda s: s, priority=30)
        plugins = self.registry.list_plugins()
        assert [p["name"] for p in plugins] == ["a", "b", "c"]

    def test_apply_modifies_segment(self):
        def uppercase(seg):
            seg["text"] = seg["text"].upper()
            return seg

        self.registry.add("upper", uppercase)
        seg = {"start": "0.000", "end": "1.000", "text": "hello world"}
        result = self.registry.apply(seg)
        assert result["text"] == "HELLO WORLD"

    def test_apply_respects_priority_order(self):
        results = []

        def plugin_a(seg):
            results.append("a")
            return seg

        def plugin_b(seg):
            results.append("b")
            return seg

        self.registry.add("b_first", plugin_b, priority=10)
        self.registry.add("a_second", plugin_a, priority=20)
        self.registry.apply({"text": "test"})
        assert results == ["b", "a"]

    def test_apply_skips_disabled(self):
        called = []

        def plugin(seg):
            called.append(True)
            return seg

        self.registry.add("disabled_plugin", plugin, enabled=False)
        self.registry.apply({"text": "test"})
        assert called == []

    def test_apply_handles_plugin_error_gracefully(self):
        def bad_plugin(seg):
            raise RuntimeError("boom")

        def good_plugin(seg):
            seg["text"] = "modified"
            return seg

        self.registry.add("bad", bad_plugin, priority=1)
        self.registry.add("good", good_plugin, priority=2)
        result = self.registry.apply({"text": "original"})
        assert result["text"] == "modified"

    def test_apply_with_none_return(self):
        def plugin_returns_none(seg):
            pass  # implicitly returns None

        self.registry.add("none_return", plugin_returns_none)
        seg = {"text": "unchanged"}
        result = self.registry.apply(seg)
        assert result["text"] == "unchanged"

    def test_clear(self):
        self.registry.add("a", lambda s: s)
        self.registry.add("b", lambda s: s)
        self.registry.clear()
        assert len(self.registry) == 0

    def test_register_decorator(self):
        @self.registry.register("decorated", priority=5)
        def my_plugin(seg):
            return seg

        assert "decorated" in self.registry
        assert self.registry._plugins["decorated"].priority == 5

    def test_chained_plugins(self):
        def add_prefix(seg):
            seg["text"] = "[PREFIX] " + seg["text"]
            return seg

        def add_suffix(seg):
            seg["text"] = seg["text"] + " [SUFFIX]"
            return seg

        self.registry.add("prefix", add_prefix, priority=1)
        self.registry.add("suffix", add_suffix, priority=2)
        result = self.registry.apply({"text": "hello"})
        assert result["text"] == "[PREFIX] hello [SUFFIX]"


class TestDefaultRegistry(unittest.TestCase):
    def test_default_registry_is_instance(self):
        assert isinstance(default_registry, PluginRegistry)


class TestServerPluginIntegration(unittest.TestCase):
    def test_server_accepts_plugin_registry(self):
        from whisper_live.server import TranscriptionServer
        server = TranscriptionServer()
        assert server.plugin_registry is None

    def test_server_sets_plugin_registry(self):
        from whisper_live.server import TranscriptionServer
        registry = PluginRegistry()
        registry.add("test", lambda s: s)
        server = TranscriptionServer()
        server.plugin_registry = registry
        assert server.plugin_registry is registry

    def test_base_client_applies_plugins(self):
        from whisper_live.backend.base import ServeClientBase
        registry = PluginRegistry()

        def uppercase(seg):
            seg["text"] = seg["text"].upper()
            return seg

        registry.add("upper", uppercase)

        client = ServeClientBase.__new__(ServeClientBase)
        client.smart_formatting = False
        client.profanity_filter = None
        client.pii_redaction = None
        client.plugin_registry = registry

        seg = client.format_segment(0.0, 1.0, "hello world")
        assert seg["text"] == "HELLO WORLD"

    def test_base_client_no_plugins(self):
        from whisper_live.backend.base import ServeClientBase

        client = ServeClientBase.__new__(ServeClientBase)
        client.smart_formatting = False
        client.profanity_filter = None
        client.pii_redaction = None
        client.plugin_registry = None

        seg = client.format_segment(0.0, 1.0, "hello world")
        assert seg["text"] == "hello world"


if __name__ == "__main__":
    unittest.main()
