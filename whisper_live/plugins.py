"""
Plugin system for WhisperLive post-processing pipeline.

Allows users to register custom post-processing plugins that run on
transcription segments. Plugins are callables that accept a segment dict
and return a modified segment dict.

Usage:
    from whisper_live.plugins import PluginRegistry

    registry = PluginRegistry()

    @registry.register("uppercase")
    def uppercase_plugin(segment):
        segment["text"] = segment["text"].upper()
        return segment

    # Or register without decorator:
    registry.add("my_plugin", my_callable, priority=10)

    # Apply all plugins to a segment
    segment = registry.apply(segment)
"""

import logging
from typing import Callable, Dict, List, Optional


class PluginEntry:
    """A registered plugin with metadata."""

    __slots__ = ("name", "fn", "priority", "enabled")

    def __init__(self, name: str, fn: Callable, priority: int = 50, enabled: bool = True):
        self.name = name
        self.fn = fn
        self.priority = priority
        self.enabled = enabled


class PluginRegistry:
    """Registry for post-processing plugins.

    Plugins are applied in priority order (lower priority numbers run first).
    Each plugin receives a segment dict and must return a segment dict.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginEntry] = {}

    def add(self, name: str, fn: Callable, priority: int = 50, enabled: bool = True):
        """Register a plugin.

        Args:
            name: Unique plugin name.
            fn: Callable that takes a segment dict and returns a segment dict.
            priority: Execution order (lower = earlier). Default 50.
            enabled: Whether the plugin is active. Default True.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' is already registered")
        self._plugins[name] = PluginEntry(name, fn, priority, enabled)

    def register(self, name: str, priority: int = 50):
        """Decorator to register a plugin function.

        Args:
            name: Unique plugin name.
            priority: Execution order (lower = earlier). Default 50.
        """
        def decorator(fn):
            self.add(name, fn, priority)
            return fn
        return decorator

    def remove(self, name: str):
        """Remove a plugin by name."""
        self._plugins.pop(name, None)

    def enable(self, name: str):
        """Enable a plugin by name."""
        if name in self._plugins:
            self._plugins[name].enabled = True

    def disable(self, name: str):
        """Disable a plugin by name."""
        if name in self._plugins:
            self._plugins[name].enabled = False

    def list_plugins(self) -> List[dict]:
        """Return list of registered plugins with their metadata."""
        return [
            {"name": p.name, "priority": p.priority, "enabled": p.enabled}
            for p in sorted(self._plugins.values(), key=lambda p: p.priority)
        ]

    def apply(self, segment: dict) -> dict:
        """Apply all enabled plugins to a segment in priority order.

        Args:
            segment: Transcription segment dict with at least 'text', 'start', 'end'.

        Returns:
            Modified segment dict after all plugins have been applied.
        """
        ordered = sorted(
            (p for p in self._plugins.values() if p.enabled),
            key=lambda p: p.priority,
        )
        for plugin in ordered:
            try:
                result = plugin.fn(segment)
                if result is not None:
                    segment = result
            except Exception as e:
                logging.error(f"Plugin '{plugin.name}' failed: {e}")
        return segment

    def clear(self):
        """Remove all registered plugins."""
        self._plugins.clear()

    def __len__(self):
        return len(self._plugins)

    def __contains__(self, name: str):
        return name in self._plugins


# Global default registry
default_registry = PluginRegistry()
