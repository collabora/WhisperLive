import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from whisper_live.utils import format_time, create_srt_file, print_transcript, clear_screen


class TestFormatTime(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(format_time(0), "00:00:00,000")

    def test_seconds_only(self):
        self.assertEqual(format_time(5.0), "00:00:05,000")

    def test_fractional_seconds(self):
        self.assertEqual(format_time(1.5), "00:00:01,500")

    def test_minutes(self):
        self.assertEqual(format_time(65.0), "00:01:05,000")

    def test_hours(self):
        self.assertEqual(format_time(3661.123), "01:01:01,123")

    def test_millisecond_precision(self):
        self.assertEqual(format_time(0.001), "00:00:00,001")

    def test_large_value(self):
        # float precision: int((86399.999 - 86399) * 1000) may be 998 or 999
        result = format_time(86399.999)
        self.assertIn(result, ("23:59:59,998", "23:59:59,999"))

    def test_rounding_edge(self):
        result = format_time(0.9999)
        # 0.9999 -> int(s%60)=0, milliseconds=int(0.9999*1000)=999
        self.assertEqual(result, "00:00:00,999")


class TestCreateSrtFile(unittest.TestCase):
    def test_single_segment(self):
        segments = [{"start": "0.000", "end": "1.500", "text": "Hello world"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            path = f.name
        try:
            create_srt_file(segments, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("1\n", content)
            self.assertIn("00:00:00,000 --> 00:00:01,500", content)
            self.assertIn("Hello world", content)
        finally:
            os.remove(path)

    def test_multiple_segments(self):
        segments = [
            {"start": "0.000", "end": "1.000", "text": "First"},
            {"start": "1.000", "end": "2.500", "text": "Second"},
            {"start": "2.500", "end": "4.000", "text": "Third"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            path = f.name
        try:
            create_srt_file(segments, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("1\n", content)
            self.assertIn("2\n", content)
            self.assertIn("3\n", content)
            self.assertIn("First", content)
            self.assertIn("Third", content)
        finally:
            os.remove(path)

    def test_empty_segments(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            path = f.name
        try:
            create_srt_file([], path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "")
        finally:
            os.remove(path)

    def test_unicode_text(self):
        segments = [{"start": "0.000", "end": "1.000", "text": "日本語テスト"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            path = f.name
        try:
            create_srt_file(segments, path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("日本語テスト", content)
        finally:
            os.remove(path)


class TestPrintTranscript(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_clear_screen_uses_ansi(self, mock_stdout):
        clear_screen()
        output = mock_stdout.getvalue()
        self.assertIn("\033[H\033[2J", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_plain_text(self, mock_stdout):
        text = ["Hello", " world"]
        print_transcript(text)
        output = mock_stdout.getvalue()
        self.assertIn("Hello world", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_with_timestamps(self, mock_stdout):
        text = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        print_transcript(text, timestamps=True)
        output = mock_stdout.getvalue()
        self.assertIn("[0.0 -> 1.0]", output)
        self.assertIn("Hello", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_translated(self, mock_stdout):
        text = ["Bonjour", "le monde"]
        print_transcript(text, translated=True)
        output = mock_stdout.getvalue()
        self.assertIn("Bonjour le monde", output)

    @patch("sys.stdout", new_callable=StringIO)
    def test_print_empty(self, mock_stdout):
        print_transcript([])
        output = mock_stdout.getvalue()
        # empty text joined is empty string, should not crash
        self.assertEqual(output.strip(), "")


if __name__ == "__main__":
    unittest.main()
