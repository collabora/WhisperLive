"""Tests for utterance detection and paragraph segmentation."""
import unittest

from whisper_live.utterance import (
    Utterance,
    Paragraph,
    UtteranceDetector,
    ParagraphSegmenter,
    detect_utterances_from_segments,
    segment_into_paragraphs,
)


class TestUtteranceDetector(unittest.TestCase):
    def test_single_segment(self):
        detector = UtteranceDetector()
        result = detector.process_segment("Hello world.", 0.0, 1.5)
        self.assertEqual(len(result), 0)  # buffered, not yet emitted
        final = detector.flush()
        self.assertIsNotNone(final)
        self.assertEqual(final.text, "Hello world.")

    def test_pause_breaks_utterance(self):
        detector = UtteranceDetector(min_pause_seconds=0.5)
        r1 = detector.process_segment("Hello.", 0.0, 1.0)
        self.assertEqual(len(r1), 0)
        # Gap of 1.0s > 0.5s threshold
        r2 = detector.process_segment("World.", 2.0, 3.0)
        self.assertEqual(len(r2), 1)
        self.assertEqual(r2[0].text, "Hello.")
        final = detector.flush()
        self.assertEqual(final.text, "World.")

    def test_sentence_end_with_short_pause(self):
        detector = UtteranceDetector(
            min_pause_seconds=1.0,
            sentence_end_pause_seconds=0.3,
        )
        r1 = detector.process_segment("How are you?", 0.0, 1.0)
        self.assertEqual(len(r1), 0)
        # Gap of 0.4s > sentence_end_pause (0.3) and text ends with ?
        r2 = detector.process_segment("I am fine.", 1.4, 2.5)
        self.assertEqual(len(r2), 1)
        self.assertEqual(r2[0].text, "How are you?")

    def test_no_break_within_sentence(self):
        detector = UtteranceDetector(min_pause_seconds=0.7)
        r1 = detector.process_segment("The quick brown", 0.0, 1.0)
        # Gap of 0.2s < 0.7s, no sentence end
        r2 = detector.process_segment("fox jumped.", 1.2, 2.0)
        self.assertEqual(len(r2), 0)
        final = detector.flush()
        self.assertEqual(final.text, "The quick brown fox jumped.")

    def test_speaker_change_breaks(self):
        detector = UtteranceDetector()
        detector.process_segment("Hello.", 0.0, 1.0, speaker="A")
        r = detector.process_segment("Hi there.", 1.1, 2.0, speaker="B")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].speaker, "A")

    def test_max_duration_breaks(self):
        detector = UtteranceDetector(max_utterance_seconds=3.0, min_pause_seconds=100)
        detector.process_segment("Long sentence", 0.0, 1.0)
        r = detector.process_segment("keeps going", 1.1, 4.0)
        self.assertEqual(len(r), 1)

    def test_reset(self):
        detector = UtteranceDetector()
        detector.process_segment("Hello.", 0.0, 1.0)
        detector.reset()
        self.assertIsNone(detector.flush())


class TestParagraphSegmenter(unittest.TestCase):
    def test_empty(self):
        seg = ParagraphSegmenter()
        self.assertEqual(seg.segment([]), [])

    def test_single_utterance(self):
        utts = [Utterance("Hello.", 0.0, 1.0)]
        paras = ParagraphSegmenter().segment(utts)
        self.assertEqual(len(paras), 1)
        self.assertEqual(paras[0].text, "Hello.")

    def test_pause_creates_new_paragraph(self):
        utts = [
            Utterance("First sentence.", 0.0, 1.0),
            Utterance("Second sentence.", 1.2, 2.0),
            # 3s gap
            Utterance("New paragraph.", 5.0, 6.0),
        ]
        paras = ParagraphSegmenter(paragraph_pause_seconds=2.0).segment(utts)
        self.assertEqual(len(paras), 2)
        self.assertEqual(len(paras[0].utterances), 2)
        self.assertEqual(len(paras[1].utterances), 1)

    def test_speaker_change_creates_paragraph(self):
        utts = [
            Utterance("Hello.", 0.0, 1.0, speaker="A"),
            Utterance("Hi.", 1.1, 2.0, speaker="B"),
        ]
        paras = ParagraphSegmenter(split_on_speaker_change=True).segment(utts)
        self.assertEqual(len(paras), 2)
        self.assertEqual(paras[0].speaker, "A")
        self.assertEqual(paras[1].speaker, "B")

    def test_max_sentences(self):
        utts = [Utterance(f"Sentence {i}.", float(i), float(i) + 0.5)
                for i in range(12)]
        paras = ParagraphSegmenter(
            paragraph_pause_seconds=100,  # won't trigger on pause
            max_sentences_per_paragraph=5,
        ).segment(utts)
        self.assertEqual(len(paras), 3)  # 5 + 5 + 2
        self.assertEqual(len(paras[0].utterances), 5)
        self.assertEqual(len(paras[1].utterances), 5)
        self.assertEqual(len(paras[2].utterances), 2)


class TestConvenienceFunctions(unittest.TestCase):
    def test_detect_utterances_from_segments(self):
        segments = [
            {"text": "Hello there.", "start": 0.0, "end": 1.0},
            {"text": "How are you?", "start": 2.0, "end": 3.0},  # 1s gap
            {"text": "I am fine.", "start": 3.2, "end": 4.0},
        ]
        utts = detect_utterances_from_segments(segments, min_pause=0.7)
        self.assertEqual(len(utts), 2)
        self.assertEqual(utts[0].text, "Hello there.")
        self.assertIn("How are you?", utts[1].text)

    def test_segment_into_paragraphs(self):
        segments = [
            {"text": "Hello.", "start": 0.0, "end": 1.0},
            {"text": "World.", "start": 1.2, "end": 2.0},
            {"text": "New topic.", "start": 5.0, "end": 6.0},
        ]
        paras = segment_into_paragraphs(segments)
        self.assertGreaterEqual(len(paras), 1)
        # All paragraphs should have text
        for p in paras:
            self.assertIn("text", p)
            self.assertIn("sentences", p)


class TestUtteranceModel(unittest.TestCase):
    def test_to_dict(self):
        u = Utterance("Hello.", 0.0, 1.5, speaker="A")
        d = u.to_dict()
        self.assertEqual(d["text"], "Hello.")
        self.assertEqual(d["speaker"], "A")
        self.assertAlmostEqual(d["start"], 0.0)

    def test_duration(self):
        u = Utterance("Test.", 1.0, 3.5)
        self.assertAlmostEqual(u.duration, 2.5)


class TestParagraphModel(unittest.TestCase):
    def test_to_dict(self):
        p = Paragraph(
            utterances=[
                Utterance("Hello.", 0.0, 1.0),
                Utterance("World.", 1.2, 2.0),
            ],
            speaker="A",
        )
        d = p.to_dict()
        self.assertEqual(d["text"], "Hello. World.")
        self.assertEqual(d["num_sentences"], 2)
        self.assertEqual(d["speaker"], "A")
        self.assertAlmostEqual(d["start"], 0.0)
        self.assertAlmostEqual(d["end"], 2.0)


if __name__ == "__main__":
    unittest.main()
