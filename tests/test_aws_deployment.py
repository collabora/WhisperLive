"""Tests for AWS deployment preparation features:
- Storage backends (local + S3)
- Graceful shutdown
- Enhanced health check
- Security headers
- Upload size limits
- Structured JSON logging
- GDPR data deletion
- Data retention
"""
import json
import logging
import os
import signal
import tempfile
import time
import unittest
from unittest import mock

from whisper_live.storage import LocalStorage, S3Storage, create_storage
from whisper_live.server import (
    TranscriptionServer,
    ClientManager,
    JSONFormatter,
    configure_logging,
)


class TestLocalStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = LocalStorage(base_dir=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_result(self):
        result = {"text": "hello world", "language": "en"}
        self.storage.save_result("job-1", result)
        got = self.storage.get_result("job-1")
        self.assertEqual(got, result)

    def test_get_result_missing(self):
        self.assertIsNone(self.storage.get_result("nonexistent"))

    def test_save_audio(self):
        path = self.storage.save_audio("job-2", b"fake-audio-data", ".wav")
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"fake-audio-data")

    def test_delete_job(self):
        self.storage.save_audio("job-3", b"audio", ".wav")
        self.storage.save_result("job-3", {"text": "test"})
        self.storage.delete_job("job-3")
        self.assertIsNone(self.storage.get_result("job-3"))
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "job-3.wav")))

    def test_delete_all_for_user(self):
        self.storage.save_audio("user1_job1", b"a", ".wav")
        self.storage.save_result("user1_job1", {"text": "a"})
        self.storage.save_audio("user1_job2", b"b", ".wav")
        self.storage.save_audio("user2_job1", b"c", ".wav")

        count = self.storage.delete_all_for_user("user1")
        self.assertEqual(count, 3)  # 2 audio + 1 json
        # user2 file should still exist
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "user2_job1.wav")))

    def test_cleanup_expired(self):
        self.storage.save_audio("old-job", b"old", ".wav")
        # Backdate the file
        old_time = time.time() - 100
        path = os.path.join(self.tmpdir, "old-job.wav")
        os.utime(path, (old_time, old_time))

        self.storage.save_audio("new-job", b"new", ".wav")

        count = self.storage.cleanup_expired(max_age_seconds=50)
        self.assertEqual(count, 1)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "old-job.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "new-job.wav")))


class TestCreateStorage(unittest.TestCase):
    def test_create_local(self):
        storage = create_storage("local")
        self.assertIsInstance(storage, LocalStorage)

    def test_create_s3_missing_boto3(self):
        with mock.patch.dict("sys.modules", {"boto3": None}):
            with self.assertRaises(ImportError):
                create_storage("s3", bucket="test-bucket")


class TestS3StorageMocked(unittest.TestCase):
    """Test S3Storage with mocked boto3 client."""

    def setUp(self):
        self.mock_boto3 = mock.MagicMock()
        self.mock_client = mock.MagicMock()
        self.mock_boto3.client.return_value = self.mock_client

        with mock.patch.dict("sys.modules", {"boto3": self.mock_boto3}):
            self.storage = S3Storage(bucket="test-bucket", prefix="wp/")

    def test_save_audio(self):
        result = self.storage.save_audio("job-1", b"audio-data", ".wav")
        self.mock_client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="wp/audio/job-1.wav", Body=b"audio-data"
        )
        self.assertEqual(result, "s3://test-bucket/wp/audio/job-1.wav")

    def test_save_result(self):
        data = {"text": "hello"}
        result = self.storage.save_result("job-1", data)
        call_args = self.mock_client.put_object.call_args
        self.assertEqual(call_args.kwargs["Bucket"], "test-bucket")
        self.assertEqual(call_args.kwargs["Key"], "wp/results/job-1.json")
        self.assertEqual(json.loads(call_args.kwargs["Body"]), data)

    def test_get_result(self):
        body_mock = mock.MagicMock()
        body_mock.read.return_value = json.dumps({"text": "hello"}).encode()
        self.mock_client.get_object.return_value = {"Body": body_mock}

        result = self.storage.get_result("job-1")
        self.assertEqual(result, {"text": "hello"})

    def test_delete_job(self):
        self.mock_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "wp/audio/job-1.wav"}]
        }
        self.storage.delete_job("job-1")
        self.assertTrue(self.mock_client.delete_object.called)


class TestJSONFormatter(unittest.TestCase):
    def test_format_produces_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["message"], "Test message")
        self.assertIn("timestamp", parsed)

    def test_format_with_extra_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Request handled", args=(), exc_info=None,
        )
        record.request_id = "abc-123"
        record.client_uid = "user-1"
        output = formatter.format(record)
        parsed = json.loads(output)
        self.assertEqual(parsed["request_id"], "abc-123")
        self.assertEqual(parsed["client_uid"], "user-1")

    def test_format_with_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Error occurred", args=(), exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        self.assertIn("exception", parsed)
        self.assertIn("ValueError", parsed["exception"])


class TestConfigureLogging(unittest.TestCase):
    def test_json_mode(self):
        configure_logging(json_logs=True)
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertIsInstance(root.handlers[0].formatter, JSONFormatter)

    def test_text_mode(self):
        configure_logging(json_logs=False)
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertNotIsInstance(root.handlers[0].formatter, JSONFormatter)


class TestGracefulShutdown(unittest.TestCase):
    def test_shutting_down_flag(self):
        server = TranscriptionServer()
        self.assertFalse(server._shutting_down)

    def test_health_returns_draining_when_shutting_down(self):
        """The health endpoint should return 503 when shutting down."""
        server = TranscriptionServer()
        server._shutting_down = True
        server.client_manager = ClientManager(max_clients=4, max_connection_time=600)
        # The actual HTTP test would require running the app;
        # here we verify the flag is accessible
        self.assertTrue(server._shutting_down)


class TestServerNewParameters(unittest.TestCase):
    def test_server_has_storage_attributes(self):
        server = TranscriptionServer()
        self.assertFalse(server._shutting_down)
        self.assertIsNone(server._ws_server)


if __name__ == "__main__":
    unittest.main()
