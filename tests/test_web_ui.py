"""Tests for web transcription UI serving."""

import os
import unittest


class TestWebUIFiles(unittest.TestCase):
    """Test that the web UI files exist and are well-formed."""

    def setUp(self):
        self.web_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "web"
        )

    def test_web_directory_exists(self):
        self.assertTrue(os.path.isdir(self.web_dir))

    def test_index_html_exists(self):
        index_path = os.path.join(self.web_dir, "index.html")
        self.assertTrue(os.path.isfile(index_path))

    def test_logo_exists(self):
        logo_path = os.path.join(self.web_dir, "Collabora_Logo.svg")
        self.assertTrue(os.path.isfile(logo_path))

    def test_index_uses_relative_api_url(self):
        index_path = os.path.join(self.web_dir, "index.html")
        with open(index_path) as f:
            content = f.read()
        self.assertIn("/v1/audio/transcriptions", content)
        self.assertNotIn("localhost:8000", content)

    def test_index_references_static_logo(self):
        index_path = os.path.join(self.web_dir, "index.html")
        with open(index_path) as f:
            content = f.read()
        self.assertIn("/static/Collabora_Logo.svg", content)

    def test_index_has_drop_zone(self):
        index_path = os.path.join(self.web_dir, "index.html")
        with open(index_path) as f:
            content = f.read()
        self.assertIn("dropZone", content)

    def test_index_has_language_selector(self):
        index_path = os.path.join(self.web_dir, "index.html")
        with open(index_path) as f:
            content = f.read()
        self.assertIn('id="language"', content)

    def test_index_has_download_button(self):
        index_path = os.path.join(self.web_dir, "index.html")
        with open(index_path) as f:
            content = f.read()
        self.assertIn("downloadBtn", content)


class TestServerWebUIMount(unittest.TestCase):
    """Test that server.py has the web UI mount code."""

    def test_server_imports_staticfiles(self):
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "whisper_live", "server.py"
        )
        with open(server_path) as f:
            content = f.read()
        self.assertIn("StaticFiles", content)

    def test_server_mounts_web_dir(self):
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "whisper_live", "server.py"
        )
        with open(server_path) as f:
            content = f.read()
        self.assertIn('app.mount("/static"', content)

    def test_server_serves_index(self):
        server_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "whisper_live", "server.py"
        )
        with open(server_path) as f:
            content = f.read()
        self.assertIn('async def serve_index', content)
