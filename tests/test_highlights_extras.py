"""Tests for auto-highlights, auto-chapters, filler removal, spelling hints, and webhooks."""
import json
import time
import unittest
from unittest import mock

from whisper_live.audio_intelligence import (
    extract_highlights,
    generate_chapters,
    remove_filler_words,
    apply_spelling_hints,
    analyze_transcript,
)
from whisper_live.webhook import send_webhook


class TestAutoHighlights(unittest.TestCase):
    def test_basic_highlights(self):
        text = (
            "Machine learning is transforming the software industry. "
            "Machine learning models are trained on large datasets. "
            "The software industry is growing rapidly with machine learning."
        )
        highlights = extract_highlights(text, max_highlights=5)
        self.assertGreater(len(highlights), 0)
        # "machine learning" should be a top phrase
        phrases = [h["text"] for h in highlights]
        self.assertTrue(any("machine" in p for p in phrases))

    def test_empty_text(self):
        self.assertEqual(extract_highlights(""), [])

    def test_highlights_have_rank(self):
        text = "Python Python Python Java Java Java code code code programming programming programming"
        highlights = extract_highlights(text, max_highlights=5)
        for h in highlights:
            self.assertIn("rank", h)
            self.assertIn("count", h)
            self.assertIn("text", h)

    def test_in_pipeline(self):
        text = "The project uses machine learning for data analysis. Machine learning is great."
        result = analyze_transcript(text, sentiment=False, topics=False,
                                     entities=False, summary=False, highlights=True)
        self.assertIn("highlights", result)


class TestAutoChapters(unittest.TestCase):
    def test_single_chapter(self):
        segments = [
            {"text": "Hello world.", "start": 0.0, "end": 1.0},
            {"text": "How are you.", "start": 1.2, "end": 2.0},
        ]
        chapters = generate_chapters(segments)
        self.assertEqual(len(chapters), 1)
        self.assertIn("title", chapters[0])
        self.assertIn("text", chapters[0])

    def test_multiple_chapters(self):
        # 3-second gap should trigger a chapter break
        segments = [
            {"text": "First topic discussion here.", "start": 0.0, "end": 30.0},
            {"text": "Still first topic.", "start": 30.5, "end": 40.0},
            {"text": "New topic starts now.", "start": 50.0, "end": 80.0},
            {"text": "Continuing new topic.", "start": 80.5, "end": 90.0},
        ]
        chapters = generate_chapters(segments, min_chapter_duration=20.0)
        self.assertGreaterEqual(len(chapters), 2)

    def test_empty(self):
        self.assertEqual(generate_chapters([]), [])

    def test_chapter_has_summary(self):
        segments = [{"text": f"Sentence number {i}." * 5, "start": float(i * 10), "end": float(i * 10 + 9)}
                    for i in range(5)]
        chapters = generate_chapters(segments, max_chapter_duration=1000)
        for ch in chapters:
            self.assertIn("summary", ch)
            self.assertIn("start", ch)
            self.assertIn("end", ch)


class TestFillerWordRemoval(unittest.TestCase):
    def test_basic_fillers(self):
        text = "So um I think uh this is like a good idea"
        result = remove_filler_words(text)
        self.assertNotIn(" um ", result)
        self.assertNotIn(" uh ", result)
        # "like" should remain in non-aggressive mode
        self.assertIn("like", result)

    def test_aggressive_mode(self):
        text = "So like basically I actually literally just wanted to say"
        result = remove_filler_words(text, aggressive=True)
        self.assertNotIn("like", result.lower())
        self.assertNotIn("basically", result.lower())

    def test_multi_word_fillers(self):
        text = "I mean it was you know pretty good sort of"
        result = remove_filler_words(text)
        self.assertNotIn("I mean", result)
        self.assertNotIn("you know", result)

    def test_empty(self):
        self.assertEqual(remove_filler_words(""), "")

    def test_preserves_real_content(self):
        text = "The meeting was productive and the results were excellent"
        result = remove_filler_words(text)
        self.assertEqual(result, text)

    def test_capitalization_preserved(self):
        text = "Um the project is going well"
        result = remove_filler_words(text)
        self.assertTrue(result[0].isupper())


class TestSpellingHints(unittest.TestCase):
    def test_basic_correction(self):
        text = "We use cube ernestes for container orchestration"
        hints = {"cube ernestes": "Kubernetes"}
        result = apply_spelling_hints(text, hints)
        self.assertIn("Kubernetes", result)
        self.assertNotIn("cube ernestes", result)

    def test_case_insensitive(self):
        text = "Install PIE TORCH on your machine"
        hints = {"pie torch": "PyTorch"}
        result = apply_spelling_hints(text, hints)
        self.assertIn("PyTorch", result)

    def test_multiple_hints(self):
        text = "We use pie torch and tensor flow for deep learning"
        hints = {"pie torch": "PyTorch", "tensor flow": "TensorFlow"}
        result = apply_spelling_hints(text, hints)
        self.assertIn("PyTorch", result)
        self.assertIn("TensorFlow", result)

    def test_empty(self):
        self.assertEqual(apply_spelling_hints("", {}), "")
        self.assertEqual(apply_spelling_hints("hello", {}), "hello")


class TestWebhookRetry(unittest.TestCase):
    @mock.patch("whisper_live.webhook.urllib.request.urlopen")
    def test_successful_delivery(self, mock_urlopen):
        mock_response = mock.MagicMock()
        mock_response.getcode.return_value = 200
        mock_urlopen.return_value = mock_response

        result = send_webhook("http://example.com/hook", {"text": "hello"})
        self.assertTrue(result)
        mock_urlopen.assert_called_once()

    @mock.patch("whisper_live.webhook.urllib.request.urlopen")
    def test_retry_on_server_error(self, mock_urlopen):
        import urllib.error
        # Fail twice with 500, succeed on third
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("http://x", 500, "err", {}, None),
            urllib.error.HTTPError("http://x", 500, "err", {}, None),
            mock.MagicMock(getcode=lambda: 200),
        ]
        result = send_webhook(
            "http://example.com/hook",
            {"text": "hello"},
            max_retries=2,
            initial_delay=0.01,
        )
        self.assertTrue(result)
        self.assertEqual(mock_urlopen.call_count, 3)

    @mock.patch("whisper_live.webhook.urllib.request.urlopen")
    def test_no_retry_on_4xx(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError("http://x", 400, "bad", {}, None)
        result = send_webhook(
            "http://example.com/hook",
            {"text": "hello"},
            max_retries=3,
            initial_delay=0.01,
        )
        self.assertFalse(result)
        mock_urlopen.assert_called_once()  # No retries for 4xx

    @mock.patch("whisper_live.webhook.urllib.request.urlopen")
    def test_retry_on_429(self, mock_urlopen):
        import urllib.error
        # 429 should be retried
        mock_urlopen.side_effect = [
            urllib.error.HTTPError("http://x", 429, "rate limited", {}, None),
            mock.MagicMock(getcode=lambda: 200),
        ]
        result = send_webhook(
            "http://example.com/hook",
            {"text": "hello"},
            max_retries=1,
            initial_delay=0.01,
        )
        self.assertTrue(result)

    @mock.patch("whisper_live.webhook.urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = send_webhook(
            "http://example.com/hook",
            {"text": "hello"},
            max_retries=2,
            initial_delay=0.01,
        )
        self.assertFalse(result)
        self.assertEqual(mock_urlopen.call_count, 3)  # 1 + 2 retries


if __name__ == "__main__":
    unittest.main()
