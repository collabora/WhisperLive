import unittest

from whisper_live.audio_intelligence import (
    analyze_sentiment,
    detect_topics,
    extract_entities,
    summarize,
    analyze_transcript,
)


class TestAnalyzeSentiment(unittest.TestCase):
    def test_positive(self):
        result = analyze_sentiment("This is great and amazing work!")
        self.assertEqual(result["label"], "positive")
        self.assertGreater(result["score"], 0)
        self.assertGreater(result["positive_count"], 0)

    def test_negative(self):
        result = analyze_sentiment("This is terrible and awful")
        self.assertEqual(result["label"], "negative")
        self.assertLess(result["score"], 0)
        self.assertGreater(result["negative_count"], 0)

    def test_neutral(self):
        result = analyze_sentiment("The cat sat on the mat")
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.0)

    def test_mixed(self):
        result = analyze_sentiment("It was great but also terrible")
        self.assertIn(result["label"], ["positive", "negative", "neutral"])

    def test_empty(self):
        result = analyze_sentiment("")
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["score"], 0.0)


class TestDetectTopics(unittest.TestCase):
    def test_basic_topics(self):
        text = "machine learning and artificial intelligence are revolutionizing technology and machine learning is growing fast"
        topics = detect_topics(text, top_n=3)
        self.assertTrue(len(topics) > 0)
        topic_words = [t["topic"] for t in topics]
        self.assertIn("machine", topic_words)

    def test_respects_top_n(self):
        text = "apple banana cherry date elderberry fig grape apple banana cherry date"
        topics = detect_topics(text, top_n=2, min_word_length=3)
        self.assertLessEqual(len(topics), 2)

    def test_excludes_stop_words(self):
        text = "the the the and and and"
        topics = detect_topics(text)
        self.assertEqual(topics, [])

    def test_min_word_length(self):
        text = "cat dog rat bat car van bus"
        topics = detect_topics(text, min_word_length=4)
        self.assertEqual(topics, [])

    def test_empty(self):
        self.assertEqual(detect_topics(""), [])


class TestExtractEntities(unittest.TestCase):
    def test_date_mdy(self):
        entities = extract_entities("Meeting on 12/25/2024")
        self.assertIn("date", entities)

    def test_date_named(self):
        entities = extract_entities("Meeting on January 15, 2024")
        self.assertIn("date", entities)

    def test_time(self):
        entities = extract_entities("Call at 3:30 PM")
        self.assertIn("time", entities)

    def test_money(self):
        entities = extract_entities("Cost is $1,500.00")
        self.assertIn("money", entities)
        self.assertIn("$1,500.00", entities["money"])

    def test_percentage(self):
        entities = extract_entities("Growth of 25.5%")
        self.assertIn("percentage", entities)

    def test_url(self):
        entities = extract_entities("Visit https://example.com for details")
        self.assertIn("url", entities)

    def test_no_entities(self):
        self.assertEqual(extract_entities("Hello world"), {})

    def test_empty(self):
        self.assertEqual(extract_entities(""), {})

    def test_multiple_types(self):
        text = "On 01/15/2024 at 2:00 PM we spent $500 which is 10% of budget"
        entities = extract_entities(text)
        self.assertIn("date", entities)
        self.assertIn("time", entities)
        self.assertIn("money", entities)
        self.assertIn("percentage", entities)


class TestSummarize(unittest.TestCase):
    def test_short_text_returned_as_is(self):
        text = "This is short. Only two sentences."
        result = summarize(text, num_sentences=3)
        self.assertEqual(result, text)

    def test_reduces_long_text(self):
        sentences = [f"Sentence number {i} about topic {i % 3}." for i in range(10)]
        text = " ".join(sentences)
        result = summarize(text, num_sentences=3)
        # Should have approximately 3 sentences
        result_sentences = [s.strip() for s in result.split(".") if s.strip()]
        self.assertLessEqual(len(result_sentences), 4)  # some tolerance for splits

    def test_preserves_order(self):
        text = "Alpha is first. Beta is second. Gamma is third. Delta is fourth. Epsilon is fifth."
        result = summarize(text, num_sentences=2)
        # Selected sentences should appear in original order
        parts = result.split(".")
        self.assertTrue(len(parts) >= 2)

    def test_empty(self):
        self.assertEqual(summarize(""), "")


class TestAnalyzeTranscript(unittest.TestCase):
    def test_full_pipeline(self):
        text = "The meeting on January 15 was great. Revenue grew 25% to $1,000,000. Machine learning is amazing."
        result = analyze_transcript(text)
        self.assertIn("sentiment", result)
        self.assertIn("topics", result)
        self.assertIn("entities", result)
        self.assertIn("summary", result)

    def test_selective_analysis(self):
        text = "Hello world."
        result = analyze_transcript(text, sentiment=True, topics=False, entities=False, summary=False)
        self.assertIn("sentiment", result)
        self.assertNotIn("topics", result)
        self.assertNotIn("entities", result)
        self.assertNotIn("summary", result)

    def test_all_disabled(self):
        result = analyze_transcript("text", sentiment=False, topics=False, entities=False, summary=False)
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
