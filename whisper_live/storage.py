"""
Pluggable storage backends for WhisperLive.

Supports local filesystem (default) and S3 for production AWS deployments.
Audio files and transcription results can be persisted for auditing,
data retention, and GDPR compliance.
"""

import os
import json
import time
import logging
import tempfile
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


class LocalStorage:
    """Store files on the local filesystem (default for dev/testing)."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or tempfile.mkdtemp(prefix="whisperlive-")
        os.makedirs(self.base_dir, exist_ok=True)

    def save_audio(self, job_id: str, data: bytes, suffix: str = ".wav") -> str:
        path = os.path.join(self.base_dir, f"{job_id}{suffix}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def save_result(self, job_id: str, result: dict) -> str:
        path = os.path.join(self.base_dir, f"{job_id}.json")
        with open(path, "w") as f:
            json.dump(result, f)
        return path

    def get_result(self, job_id: str) -> Optional[dict]:
        path = os.path.join(self.base_dir, f"{job_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def delete_job(self, job_id: str):
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".json"]:
            path = os.path.join(self.base_dir, f"{job_id}{ext}")
            if os.path.exists(path):
                os.unlink(path)

    def delete_all_for_user(self, user_id: str) -> int:
        """GDPR: delete all files for a given user. Requires user_id prefix convention."""
        count = 0
        prefix = f"{user_id}_"
        for fname in os.listdir(self.base_dir):
            if fname.startswith(prefix):
                os.unlink(os.path.join(self.base_dir, fname))
                count += 1
        return count

    def cleanup_expired(self, max_age_seconds: int) -> int:
        """Delete files older than max_age_seconds. Returns count of deleted files."""
        count = 0
        now = time.time()
        for fname in os.listdir(self.base_dir):
            fpath = os.path.join(self.base_dir, fname)
            if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_seconds:
                os.unlink(fpath)
                count += 1
        return count


class S3Storage:
    """Store files in AWS S3 for production deployments."""

    def __init__(self, bucket: str, prefix: str = "whisperlive/", region: Optional[str] = None):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
        self.bucket = bucket
        self.prefix = prefix
        kwargs = {}
        if region:
            kwargs["region_name"] = region
        # Support S3-compatible endpoints (MinIO, LocalStack, etc.)
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        self._s3 = boto3.client("s3", **kwargs)
        logger.info(f"S3 storage initialized: s3://{bucket}/{prefix}"
                    + (f" (endpoint: {endpoint_url})" if endpoint_url else ""))

    def _key(self, filename: str) -> str:
        return f"{self.prefix}{filename}"

    def save_audio(self, job_id: str, data: bytes, suffix: str = ".wav") -> str:
        key = self._key(f"audio/{job_id}{suffix}")
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def save_result(self, job_id: str, result: dict) -> str:
        key = self._key(f"results/{job_id}.json")
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(result).encode(),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{key}"

    def get_result(self, job_id: str) -> Optional[dict]:
        key = self._key(f"results/{job_id}.json")
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get result for {job_id}: {e}")
            return None

    def delete_job(self, job_id: str):
        for prefix_path in ["audio/", "results/"]:
            # List objects with the job_id prefix
            resp = self._s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self._key(f"{prefix_path}{job_id}"),
            )
            for obj in resp.get("Contents", []):
                self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def delete_all_for_user(self, user_id: str) -> int:
        """GDPR: delete all S3 objects for a given user."""
        count = 0
        paginator = self._s3.get_paginator("list_objects_v2")
        for prefix_path in ["audio/", "results/"]:
            for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=self._key(f"{prefix_path}{user_id}_"),
            ):
                for obj in page.get("Contents", []):
                    self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])
                    count += 1
        return count

    def cleanup_expired(self, max_age_seconds: int) -> int:
        """
        Delete S3 objects older than max_age_seconds.
        Note: For production, prefer S3 Lifecycle Rules configured on the bucket.
        """
        count = 0
        now = time.time()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                age = now - obj["LastModified"].timestamp()
                if age > max_age_seconds:
                    self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])
                    count += 1
        return count


def create_storage(backend: str = "local", **kwargs):
    """Factory to create the appropriate storage backend.

    Args:
        backend: "local" or "s3"
        **kwargs: Passed to the storage class constructor.
            For S3: bucket (required), prefix, region
            For local: base_dir
    """
    if backend == "s3":
        return S3Storage(**kwargs)
    return LocalStorage(**kwargs)
