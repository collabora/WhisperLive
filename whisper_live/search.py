"""
Transcript search and tagging for stored transcription results.

Provides:
- Full-text search across stored transcription results
- Tagging/metadata on transcription jobs
- Usage tracking and billing-style API endpoints
"""

import re
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class TranscriptMetadata:
    """Metadata for a stored transcription job."""
    job_id: str
    user_id: str = ""
    created_at: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    language: str = ""
    model: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    text: str = ""  # Full transcript text for search

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptMetadata":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TranscriptIndex:
    """In-memory index for transcript search and tagging.

    Designed to work alongside the storage backend. The storage backend
    handles audio/result persistence; this index handles search and metadata.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._index: Dict[str, TranscriptMetadata] = {}

    def add(self, metadata: TranscriptMetadata):
        """Add or update a transcript in the index."""
        with self._lock:
            self._index[metadata.job_id] = metadata

    def get(self, job_id: str) -> Optional[TranscriptMetadata]:
        """Get metadata for a specific job."""
        with self._lock:
            return self._index.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Remove a job from the index."""
        with self._lock:
            return self._index.pop(job_id, None) is not None

    def tag(self, job_id: str, tags: Dict[str, str]) -> bool:
        """Add or update tags on a job.

        Args:
            job_id: The job ID.
            tags: Dict of tag key-value pairs to add/update.

        Returns:
            True if job exists and was tagged.
        """
        with self._lock:
            meta = self._index.get(job_id)
            if not meta:
                return False
            meta.tags.update(tags)
            return True

    def untag(self, job_id: str, keys: List[str]) -> bool:
        """Remove specific tags from a job."""
        with self._lock:
            meta = self._index.get(job_id)
            if not meta:
                return False
            for key in keys:
                meta.tags.pop(key, None)
            return True

    def search(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        language: Optional[str] = None,
        model: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TranscriptMetadata]:
        """Search transcripts by text content and/or metadata filters.

        Args:
            query: Full-text search query (case-insensitive substring match).
            user_id: Filter by user ID.
            tags: Filter by tags (all must match).
            language: Filter by detected language.
            model: Filter by model used.
            start_time: Filter by created_at >= start_time (epoch).
            end_time: Filter by created_at <= end_time (epoch).
            limit: Maximum results to return.
            offset: Pagination offset.

        Returns:
            List of matching TranscriptMetadata.
        """
        with self._lock:
            results = []
            query_lower = query.lower() if query else ""

            for meta in self._index.values():
                # Text search
                if query_lower and query_lower not in meta.text.lower():
                    continue

                # User filter
                if user_id and meta.user_id != user_id:
                    continue

                # Tag filter (all must match)
                if tags:
                    if not all(meta.tags.get(k) == v for k, v in tags.items()):
                        continue

                # Language filter
                if language and meta.language != language:
                    continue

                # Model filter
                if model and meta.model != model:
                    continue

                # Time range
                if start_time and meta.created_at < start_time:
                    continue
                if end_time and meta.created_at > end_time:
                    continue

                results.append(meta)

            # Sort by creation time, newest first
            results.sort(key=lambda m: m.created_at, reverse=True)
            return results[offset:offset + limit]

    def count(self, user_id: Optional[str] = None) -> int:
        """Count total transcripts, optionally filtered by user."""
        with self._lock:
            if user_id:
                return sum(1 for m in self._index.values() if m.user_id == user_id)
            return len(self._index)

    def list_all(self, limit: int = 50, offset: int = 0) -> List[TranscriptMetadata]:
        """List all transcripts with pagination."""
        return self.search(limit=limit, offset=offset)


@dataclass
class UsagePeriod:
    """Usage statistics for a time period."""
    user_id: str
    period_start: float
    period_end: float
    total_requests: int = 0
    total_audio_minutes: float = 0.0
    total_characters: int = 0
    by_model: Dict[str, int] = field(default_factory=dict)
    by_language: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class UsageTracker:
    """Track API usage for billing and analytics.

    Stores usage data in memory. For production, persist to the storage backend.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # user_id -> list of usage events
        self._events: Dict[str, List[dict]] = {}

    def record(
        self,
        user_id: str,
        audio_minutes: float = 0.0,
        characters: int = 0,
        model: str = "",
        language: str = "",
    ):
        """Record a usage event."""
        event = {
            "timestamp": time.time(),
            "audio_minutes": audio_minutes,
            "characters": characters,
            "model": model,
            "language": language,
        }
        with self._lock:
            self._events.setdefault(user_id, []).append(event)

    def get_usage(
        self,
        user_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> UsagePeriod:
        """Get usage statistics for a user in a time period.

        Args:
            user_id: The user to query.
            start_time: Start of period (epoch). Default: beginning of current month.
            end_time: End of period (epoch). Default: now.
        """
        import calendar
        import datetime

        now = time.time()
        if start_time is None:
            # Default: beginning of current month
            dt = datetime.datetime.now()
            start_time = datetime.datetime(dt.year, dt.month, 1).timestamp()
        if end_time is None:
            end_time = now

        with self._lock:
            events = self._events.get(user_id, [])

        period = UsagePeriod(
            user_id=user_id,
            period_start=start_time,
            period_end=end_time,
        )

        for event in events:
            if event["timestamp"] < start_time or event["timestamp"] > end_time:
                continue
            period.total_requests += 1
            period.total_audio_minutes += event["audio_minutes"]
            period.total_characters += event["characters"]
            model = event.get("model", "unknown")
            period.by_model[model] = period.by_model.get(model, 0) + 1
            lang = event.get("language", "unknown")
            period.by_language[lang] = period.by_language.get(lang, 0) + 1

        return period

    def get_all_usage(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[UsagePeriod]:
        """Get usage for all users."""
        with self._lock:
            user_ids = list(self._events.keys())
        return [self.get_usage(uid, start_time, end_time) for uid in user_ids]
