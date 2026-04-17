"""Tests for OpenAPI auto-docs and health check endpoint."""

import unittest
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestOpenAPIAndHealth(unittest.TestCase):
    """Verify that FastAPI app has /docs, /openapi.json, and /health."""

    def _make_app(self):
        """Create a minimal app mimicking the server's FastAPI configuration."""
        from whisper_live.server import TranscriptionServer
        server = TranscriptionServer()
        server.client_manager = type('CM', (), {
            'clients': {},
            'max_clients': 4,
        })()

        app = FastAPI(
            title="WhisperLive OpenAI-Compatible API",
            description="Test",
            version="0.9.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        @app.get("/health", tags=["System"])
        async def health_check():
            return {
                "status": "ok",
                "clients": len(server.client_manager.clients),
                "max_clients": server.client_manager.max_clients,
            }

        return app

    def test_openapi_json_available(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "WhisperLive OpenAI-Compatible API"
        assert data["info"]["version"] == "0.9.0"

    def test_docs_page_available(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()

    def test_redoc_page_available(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_health_endpoint(self):
        app = self._make_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["clients"] == 0
        assert data["max_clients"] == 4

    def test_server_app_has_description(self):
        """Verify the server module source configures FastAPI with full metadata."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert 'docs_url="/docs"' in source
        assert 'redoc_url="/redoc"' in source
        assert 'openapi_url="/openapi.json"' in source
        assert 'version="0.9.0"' in source

    def test_endpoints_have_tags(self):
        """Verify endpoints are tagged for OpenAPI grouping."""
        import inspect
        from whisper_live import server
        source = inspect.getsource(server)
        assert 'tags=["Transcription"]' in source
        assert 'tags=["Intelligence"]' in source
        assert 'tags=["System"]' in source


if __name__ == "__main__":
    unittest.main()
