"""Tests for multi-model ensemble transcription."""

import unittest
from unittest.mock import MagicMock

from whisper_live.ensemble import (
    EnsembleTranscriber,
    EnsembleResult,
    Segment,
)


def _make_segments(texts, confidence=0.9, model_name="test"):
    """Helper to create segments from a list of text strings."""
    segs = []
    t = 0.0
    for text in texts:
        segs.append(Segment(start=t, end=t + 1.0, text=text, confidence=confidence, model_name=model_name))
        t += 1.0
    return segs


class TestEnsembleTranscriber(unittest.TestCase):
    """Tests for EnsembleTranscriber."""

    def test_empty_models_returns_empty(self):
        ens = EnsembleTranscriber(models={})
        result = ens.transcribe(b"")
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])

    def test_single_model(self):
        segs = _make_segments(["hello", "world"])
        fn = MagicMock(return_value=segs)
        ens = EnsembleTranscriber(models={"m1": fn}, strategy="longest")
        result = ens.transcribe(b"audio")
        fn.assert_called_once_with(b"audio")
        self.assertIn("hello", result.text)
        self.assertIn("world", result.text)

    def test_add_remove_model(self):
        ens = EnsembleTranscriber()
        fn = MagicMock(return_value=[])
        ens.add_model("m1", fn)
        self.assertEqual(ens.model_names, ["m1"])
        ens.remove_model("m1")
        self.assertEqual(ens.model_names, [])

    def test_remove_nonexistent(self):
        ens = EnsembleTranscriber()
        ens.remove_model("nope")  # should not raise

    def test_strategy_property(self):
        ens = EnsembleTranscriber(strategy="longest")
        self.assertEqual(ens.strategy, "longest")
        ens.strategy = "confidence"
        self.assertEqual(ens.strategy, "confidence")

    def test_strategy_invalid(self):
        ens = EnsembleTranscriber()
        with self.assertRaises(ValueError):
            ens.strategy = "invalid"


class TestMergeLongest(unittest.TestCase):
    """Tests for the 'longest' merge strategy."""

    def test_picks_longest_output(self):
        short = _make_segments(["hi"])
        long = _make_segments(["hello there", "how are you today"])
        ens = EnsembleTranscriber(
            models={
                "short": MagicMock(return_value=short),
                "long": MagicMock(return_value=long),
            },
            strategy="longest",
        )
        result = ens.transcribe(b"audio")
        self.assertIn("hello there", result.text)
        self.assertIn("how are you today", result.text)

    def test_all_empty(self):
        ens = EnsembleTranscriber(
            models={"m1": MagicMock(return_value=[])},
            strategy="longest",
        )
        result = ens.transcribe(b"audio")
        self.assertEqual(result.text, "")


class TestMergeConfidence(unittest.TestCase):
    """Tests for the 'confidence' merge strategy."""

    def test_picks_highest_confidence(self):
        low = _make_segments(["low quality"], confidence=0.3)
        high = _make_segments(["high quality"], confidence=0.95)
        ens = EnsembleTranscriber(
            models={
                "low": MagicMock(return_value=low),
                "high": MagicMock(return_value=high),
            },
            strategy="confidence",
        )
        result = ens.transcribe(b"audio")
        self.assertIn("high quality", result.text)

    def test_skips_empty(self):
        empty = []
        good = _make_segments(["data"], confidence=0.8)
        ens = EnsembleTranscriber(
            models={
                "empty": MagicMock(return_value=empty),
                "good": MagicMock(return_value=good),
            },
            strategy="confidence",
        )
        result = ens.transcribe(b"audio")
        self.assertIn("data", result.text)


class TestMergeVoting(unittest.TestCase):
    """Tests for the 'voting' merge strategy."""

    def test_majority_wins(self):
        # Two models agree on "hello", one says "hullo"
        seg_a = _make_segments(["hello"])
        seg_b = _make_segments(["hello"])
        seg_c = _make_segments(["hullo"])
        ens = EnsembleTranscriber(
            models={
                "a": MagicMock(return_value=seg_a),
                "b": MagicMock(return_value=seg_b),
                "c": MagicMock(return_value=seg_c),
            },
            strategy="voting",
        )
        result = ens.transcribe(b"audio")
        self.assertEqual(result.text, "hello")

    def test_single_input(self):
        seg = _make_segments(["only one"])
        ens = EnsembleTranscriber(
            models={"m1": MagicMock(return_value=seg)},
            strategy="voting",
        )
        result = ens.transcribe(b"audio")
        self.assertEqual(result.text, "only one")

    def test_all_empty(self):
        ens = EnsembleTranscriber(
            models={"m1": MagicMock(return_value=[])},
            strategy="voting",
        )
        result = ens.transcribe(b"audio")
        self.assertEqual(result.text, "")


class TestModelFailure(unittest.TestCase):
    """Tests for model error handling."""

    def test_failed_model_returns_empty(self):
        def fail_fn(audio):
            raise RuntimeError("model crashed")

        good = _make_segments(["survived"])
        ens = EnsembleTranscriber(
            models={
                "fail": fail_fn,
                "good": MagicMock(return_value=good),
            },
            strategy="longest",
        )
        result = ens.transcribe(b"audio")
        self.assertIn("survived", result.text)
        self.assertEqual(result.model_results["fail"], [])

    def test_all_fail(self):
        def fail_fn(audio):
            raise RuntimeError("boom")

        ens = EnsembleTranscriber(
            models={"a": fail_fn, "b": fail_fn},
            strategy="longest",
        )
        result = ens.transcribe(b"audio")
        self.assertEqual(result.text, "")


class TestEnsembleResult(unittest.TestCase):
    """Tests for EnsembleResult structure."""

    def test_model_results_populated(self):
        seg = _make_segments(["test"])
        ens = EnsembleTranscriber(
            models={"m1": MagicMock(return_value=seg)},
            strategy="longest",
        )
        result = ens.transcribe(b"audio")
        self.assertIn("m1", result.model_results)
        self.assertEqual(result.strategy, "longest")

    def test_segment_model_name_set(self):
        seg = _make_segments(["test"])
        fn = MagicMock(return_value=seg)
        ens = EnsembleTranscriber(models={"mymodel": fn}, strategy="longest")
        result = ens.transcribe(b"audio")
        for s in result.model_results["mymodel"]:
            self.assertEqual(s.model_name, "mymodel")


if __name__ == "__main__":
    unittest.main()
