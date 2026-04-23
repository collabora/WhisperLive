"""Tests for live translation relay."""

import unittest
from whisper_live.translation_relay import (
    TranslationRelay,
    RelayChannel,
    Subscriber,
)


class TestSubscriber(unittest.TestCase):
    def test_callback_delivery(self):
        received = []
        sub = Subscriber("s1", "fr", callback=lambda seg: received.append(seg))
        sub.deliver({"text": "hello"})
        assert len(received) == 1
        assert received[0]["text"] == "hello"

    def test_queue_delivery(self):
        sub = Subscriber("s1", "fr")
        sub.deliver({"text": "hello"})
        sub.deliver({"text": "world"})
        items = sub.drain()
        assert len(items) == 2
        assert len(sub.queue) == 0

    def test_callback_error_handled(self):
        def bad_callback(seg):
            raise RuntimeError("boom")

        sub = Subscriber("s1", "fr", callback=bad_callback)
        # Should not raise
        sub.deliver({"text": "hello"})


class TestRelayChannel(unittest.TestCase):
    def test_add_and_remove_subscriber(self):
        ch = RelayChannel("ch1", "en")
        ch.add_subscriber("s1", "fr")
        assert ch.info()["subscribers"] == 1
        ch.remove_subscriber("s1")
        assert ch.info()["subscribers"] == 0

    def test_broadcast_to_subscribers(self):
        received = {"s1": [], "s2": []}
        ch = RelayChannel("ch1", "en")
        ch.add_subscriber("s1", "fr", callback=lambda seg: received["s1"].append(seg))
        ch.add_subscriber("s2", "de", callback=lambda seg: received["s2"].append(seg))

        ch.broadcast({"text": "hello", "start": "0.000", "end": "1.000"})
        assert len(received["s1"]) == 1
        assert len(received["s2"]) == 1

    def test_broadcast_with_translator(self):
        def fake_translate(text, source_lang, target_lang):
            return f"[{target_lang}] {text}"

        received = []
        ch = RelayChannel("ch1", "en", translator=fake_translate)
        ch.add_subscriber("s1", "fr", callback=lambda seg: received.append(seg))

        ch.broadcast({"text": "hello", "start": "0.000", "end": "1.000"})
        assert received[0]["text"] == "[fr] hello"
        assert received[0]["original_text"] == "hello"
        assert received[0]["target_language"] == "fr"

    def test_broadcast_same_language_no_translation(self):
        def fake_translate(text, source_lang, target_lang):
            return f"[{target_lang}] {text}"

        received = []
        ch = RelayChannel("ch1", "en", translator=fake_translate)
        ch.add_subscriber("s1", "en", callback=lambda seg: received.append(seg))

        ch.broadcast({"text": "hello", "start": "0.000", "end": "1.000"})
        # Same language = no translation
        assert received[0]["text"] == "hello"

    def test_info(self):
        ch = RelayChannel("ch1", "en")
        ch.add_subscriber("s1", "fr")
        ch.add_subscriber("s2", "de")
        info = ch.info()
        assert info["channel_id"] == "ch1"
        assert info["source_language"] == "en"
        assert info["subscribers"] == 2
        assert set(info["target_languages"]) == {"fr", "de"}

    def test_close_all(self):
        ch = RelayChannel("ch1", "en")
        ch.add_subscriber("s1", "fr")
        ch.add_subscriber("s2", "de")
        ch.close_all()
        assert ch.info()["subscribers"] == 0

    def test_translator_error_handled(self):
        def bad_translate(text, source_lang, target_lang):
            raise RuntimeError("translate failed")

        received = []
        ch = RelayChannel("ch1", "en", translator=bad_translate)
        ch.add_subscriber("s1", "fr", callback=lambda seg: received.append(seg))
        ch.broadcast({"text": "hello", "start": "0.000", "end": "1.000"})
        assert "translation_error" in received[0]


class TestTranslationRelay(unittest.TestCase):
    def test_create_channel(self):
        relay = TranslationRelay()
        relay.create_channel("meeting-1", "en")
        channels = relay.list_channels()
        assert len(channels) == 1
        assert channels[0]["channel_id"] == "meeting-1"

    def test_duplicate_channel_raises(self):
        relay = TranslationRelay()
        relay.create_channel("meeting-1")
        with self.assertRaises(ValueError):
            relay.create_channel("meeting-1")

    def test_remove_channel(self):
        relay = TranslationRelay()
        relay.create_channel("meeting-1")
        relay.remove_channel("meeting-1")
        assert relay.list_channels() == []

    def test_subscribe_and_publish(self):
        received = []
        relay = TranslationRelay()
        relay.create_channel("ch1", "en")
        relay.subscribe("ch1", "fr", "listener-1",
                        callback=lambda seg: received.append(seg))
        relay.publish("ch1", {"text": "hello", "start": "0.000", "end": "1.000"})
        assert len(received) == 1

    def test_subscribe_nonexistent_channel(self):
        relay = TranslationRelay()
        assert relay.subscribe("missing", "fr", "s1") is False

    def test_unsubscribe(self):
        relay = TranslationRelay()
        relay.create_channel("ch1")
        relay.subscribe("ch1", "fr", "s1")
        relay.unsubscribe("ch1", "s1")
        ch = relay.get_channel("ch1")
        assert ch.info()["subscribers"] == 0

    def test_publish_nonexistent_channel_no_error(self):
        relay = TranslationRelay()
        relay.publish("missing", {"text": "hello"})  # should not raise

    def test_multi_language_broadcast(self):
        received_fr = []
        received_de = []

        def fake_translate(text, source_lang, target_lang):
            return f"[{target_lang}] {text}"

        relay = TranslationRelay()
        relay.create_channel("ch1", "en")
        ch = relay.get_channel("ch1")
        ch.translator = fake_translate

        relay.subscribe("ch1", "fr", "fr-listener",
                        callback=lambda seg: received_fr.append(seg))
        relay.subscribe("ch1", "de", "de-listener",
                        callback=lambda seg: received_de.append(seg))

        relay.publish("ch1", {"text": "hello", "start": "0.000", "end": "1.000"})
        assert received_fr[0]["text"] == "[fr] hello"
        assert received_de[0]["text"] == "[de] hello"


if __name__ == "__main__":
    unittest.main()
