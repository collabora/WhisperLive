"""Tests for transcript search, tagging, and usage tracking."""
import time
import unittest

from whisper_live.search import (
    TranscriptMetadata,
    TranscriptIndex,
    UsageTracker,
    UsagePeriod,
)


class TestTranscriptMetadata(unittest.TestCase):
    def test_round_trip(self):
        meta = TranscriptMetadata(
            job_id="job-1",
            user_id="user-1",
            text="Hello world",
            language="en",
            model="small",
            tags={"project": "demo"},
        )
        d = meta.to_dict()
        restored = TranscriptMetadata.from_dict(d)
        self.assertEqual(restored.job_id, "job-1")
        self.assertEqual(restored.tags["project"], "demo")


class TestTranscriptIndex(unittest.TestCase):
    def setUp(self):
        self.index = TranscriptIndex()
        self.index.add(TranscriptMetadata(
            job_id="job-1", user_id="user-1", text="The weather is sunny today",
            language="en", model="small", tags={"type": "meeting"},
            created_at=1000.0,
        ))
        self.index.add(TranscriptMetadata(
            job_id="job-2", user_id="user-1", text="Revenue increased by twenty percent",
            language="en", model="large", tags={"type": "earnings"},
            created_at=2000.0,
        ))
        self.index.add(TranscriptMetadata(
            job_id="job-3", user_id="user-2", text="Bonjour le monde",
            language="fr", model="small", tags={"type": "meeting"},
            created_at=3000.0,
        ))

    def test_search_text(self):
        results = self.index.search(query="weather")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].job_id, "job-1")

    def test_search_case_insensitive(self):
        results = self.index.search(query="REVENUE")
        self.assertEqual(len(results), 1)

    def test_search_by_user(self):
        results = self.index.search(user_id="user-1")
        self.assertEqual(len(results), 2)

    def test_search_by_language(self):
        results = self.index.search(language="fr")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].job_id, "job-3")

    def test_search_by_model(self):
        results = self.index.search(model="large")
        self.assertEqual(len(results), 1)

    def test_search_by_tags(self):
        results = self.index.search(tags={"type": "meeting"})
        self.assertEqual(len(results), 2)

    def test_search_combined_filters(self):
        results = self.index.search(query="weather", language="en", tags={"type": "meeting"})
        self.assertEqual(len(results), 1)

    def test_search_time_range(self):
        results = self.index.search(start_time=1500.0, end_time=2500.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].job_id, "job-2")

    def test_search_pagination(self):
        results = self.index.search(limit=1, offset=0)
        self.assertEqual(len(results), 1)
        results2 = self.index.search(limit=1, offset=1)
        self.assertEqual(len(results2), 1)
        self.assertNotEqual(results[0].job_id, results2[0].job_id)

    def test_search_results_sorted_newest_first(self):
        results = self.index.search()
        self.assertEqual(results[0].job_id, "job-3")  # created_at=3000
        self.assertEqual(results[-1].job_id, "job-1")  # created_at=1000

    def test_get(self):
        meta = self.index.get("job-1")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.text, "The weather is sunny today")

    def test_get_missing(self):
        self.assertIsNone(self.index.get("nonexistent"))

    def test_delete(self):
        self.assertTrue(self.index.delete("job-1"))
        self.assertIsNone(self.index.get("job-1"))
        self.assertFalse(self.index.delete("nonexistent"))

    def test_count(self):
        self.assertEqual(self.index.count(), 3)
        self.assertEqual(self.index.count(user_id="user-1"), 2)

    def test_tag(self):
        self.assertTrue(self.index.tag("job-1", {"priority": "high"}))
        meta = self.index.get("job-1")
        self.assertEqual(meta.tags["priority"], "high")
        self.assertEqual(meta.tags["type"], "meeting")  # Original still there

    def test_tag_missing(self):
        self.assertFalse(self.index.tag("nonexistent", {"a": "b"}))

    def test_untag(self):
        self.assertTrue(self.index.untag("job-1", ["type"]))
        meta = self.index.get("job-1")
        self.assertNotIn("type", meta.tags)

    def test_untag_missing(self):
        self.assertFalse(self.index.untag("nonexistent", ["type"]))


class TestUsageTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = UsageTracker()

    def test_record_and_get(self):
        self.tracker.record("user-1", audio_minutes=5.0, characters=1000, model="small", language="en")
        self.tracker.record("user-1", audio_minutes=3.0, characters=500, model="large", language="en")
        usage = self.tracker.get_usage("user-1", start_time=0)
        self.assertEqual(usage.total_requests, 2)
        self.assertAlmostEqual(usage.total_audio_minutes, 8.0)
        self.assertEqual(usage.total_characters, 1500)
        self.assertEqual(usage.by_model["small"], 1)
        self.assertEqual(usage.by_model["large"], 1)

    def test_empty_user(self):
        usage = self.tracker.get_usage("nobody", start_time=0)
        self.assertEqual(usage.total_requests, 0)

    def test_time_filtering(self):
        self.tracker.record("user-1", audio_minutes=1.0)
        usage = self.tracker.get_usage("user-1", start_time=time.time() + 100)
        self.assertEqual(usage.total_requests, 0)

    def test_get_all_usage(self):
        self.tracker.record("user-1", audio_minutes=5.0)
        self.tracker.record("user-2", audio_minutes=3.0)
        all_usage = self.tracker.get_all_usage(start_time=0)
        self.assertEqual(len(all_usage), 2)

    def test_usage_period_serialization(self):
        period = UsagePeriod(
            user_id="user-1",
            period_start=1000.0,
            period_end=2000.0,
            total_requests=5,
        )
        d = period.to_dict()
        self.assertEqual(d["user_id"], "user-1")
        self.assertEqual(d["total_requests"], 5)


if __name__ == "__main__":
    unittest.main()
