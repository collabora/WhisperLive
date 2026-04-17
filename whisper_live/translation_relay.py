"""
Live translation relay for WhisperLive.

Provides a pub/sub relay that takes transcription segments from a source
WebSocket session, translates them in real-time, and broadcasts translated
segments to subscriber WebSocket connections.

This enables multilingual meeting scenarios where one speaker is
transcribed and multiple listeners receive translations in their
preferred languages simultaneously.
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Set


class TranslationRelay:
    """Manages translation relay channels for multilingual broadcasting.

    Each channel represents a source language being translated to one
    or more target languages. Subscribers receive translated segments
    in real-time via their WebSocket connections.
    """

    def __init__(self):
        self._channels: Dict[str, RelayChannel] = {}
        self._lock = threading.Lock()

    def create_channel(self, channel_id: str, source_language: str = "en"):
        """Create a new relay channel.

        Args:
            channel_id: Unique identifier for the channel.
            source_language: Source language code (e.g. "en").

        Raises:
            ValueError: If channel already exists.
        """
        with self._lock:
            if channel_id in self._channels:
                raise ValueError(f"Channel '{channel_id}' already exists")
            self._channels[channel_id] = RelayChannel(channel_id, source_language)
            logging.info(f"Translation relay channel created: {channel_id} ({source_language})")

    def remove_channel(self, channel_id: str):
        """Remove a relay channel and disconnect all subscribers."""
        with self._lock:
            channel = self._channels.pop(channel_id, None)
        if channel:
            channel.close_all()

    def get_channel(self, channel_id: str) -> Optional["RelayChannel"]:
        """Get a channel by ID."""
        with self._lock:
            return self._channels.get(channel_id)

    def list_channels(self) -> List[dict]:
        """Return list of active channels with subscriber counts."""
        with self._lock:
            return [ch.info() for ch in self._channels.values()]

    def publish(self, channel_id: str, segment: dict):
        """Publish a transcription segment to a channel.

        Args:
            channel_id: Channel to publish to.
            segment: Transcription segment dict with 'text', 'start', 'end'.
        """
        with self._lock:
            channel = self._channels.get(channel_id)
        if channel:
            channel.broadcast(segment)

    def subscribe(self, channel_id: str, target_language: str, subscriber_id: str,
                  callback=None):
        """Subscribe to a channel for a specific target language.

        Args:
            channel_id: Channel to subscribe to.
            target_language: Desired translation language code.
            subscriber_id: Unique identifier for the subscriber.
            callback: Optional callable(segment_dict) to invoke on each
                translated segment. If None, segments are queued.

        Returns:
            bool: True if subscription succeeded.
        """
        with self._lock:
            channel = self._channels.get(channel_id)
        if not channel:
            return False
        channel.add_subscriber(subscriber_id, target_language, callback)
        return True

    def unsubscribe(self, channel_id: str, subscriber_id: str):
        """Remove a subscriber from a channel."""
        with self._lock:
            channel = self._channels.get(channel_id)
        if channel:
            channel.remove_subscriber(subscriber_id)


class Subscriber:
    """A relay channel subscriber."""

    __slots__ = ("subscriber_id", "target_language", "callback", "queue")

    def __init__(self, subscriber_id: str, target_language: str, callback=None):
        self.subscriber_id = subscriber_id
        self.target_language = target_language
        self.callback = callback
        self.queue: List[dict] = []

    def deliver(self, segment: dict):
        """Deliver a segment to this subscriber."""
        if self.callback:
            try:
                self.callback(segment)
            except Exception as e:
                logging.error(f"Subscriber '{self.subscriber_id}' callback error: {e}")
        else:
            self.queue.append(segment)

    def drain(self) -> List[dict]:
        """Return and clear queued segments."""
        items = list(self.queue)
        self.queue.clear()
        return items


class RelayChannel:
    """A single relay channel with subscribers.

    Broadcasts transcription segments to all subscribers. Translation
    is delegated to subscribers' callbacks or an external translation
    function.
    """

    def __init__(self, channel_id: str, source_language: str = "en",
                 translator=None):
        self.channel_id = channel_id
        self.source_language = source_language
        self.translator = translator
        self._subscribers: Dict[str, Subscriber] = {}
        self._lock = threading.Lock()

    def add_subscriber(self, subscriber_id: str, target_language: str,
                       callback=None):
        """Add a subscriber to this channel."""
        with self._lock:
            self._subscribers[subscriber_id] = Subscriber(
                subscriber_id, target_language, callback
            )
        logging.info(
            f"Subscriber '{subscriber_id}' joined channel '{self.channel_id}' "
            f"for {target_language}"
        )

    def remove_subscriber(self, subscriber_id: str):
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def broadcast(self, segment: dict):
        """Broadcast a segment to all subscribers.

        If a translator function is set, it translates text before delivery.
        Otherwise, sends the original segment.
        """
        with self._lock:
            subscribers = list(self._subscribers.values())

        for sub in subscribers:
            translated_segment = dict(segment)
            if self.translator and sub.target_language != self.source_language:
                try:
                    translated_segment["text"] = self.translator(
                        segment["text"],
                        source_lang=self.source_language,
                        target_lang=sub.target_language,
                    )
                    translated_segment["original_text"] = segment["text"]
                    translated_segment["target_language"] = sub.target_language
                except Exception as e:
                    logging.error(f"Translation error for subscriber '{sub.subscriber_id}': {e}")
                    translated_segment["translation_error"] = str(e)
            sub.deliver(translated_segment)

    def info(self) -> dict:
        """Return channel info."""
        with self._lock:
            return {
                "channel_id": self.channel_id,
                "source_language": self.source_language,
                "subscribers": len(self._subscribers),
                "target_languages": list({
                    s.target_language for s in self._subscribers.values()
                }),
            }

    def close_all(self):
        """Remove all subscribers."""
        with self._lock:
            self._subscribers.clear()
