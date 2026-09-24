from __future__ import annotations

from http import HTTPStatus
from io import BytesIO
import json
from types import SimpleNamespace
import unittest

from helpers import SRC_ROOT  # noqa: F401 - importing helpers bootstraps src/
from test_poc_acceptance import FakeLMStudioTransport, _adapter

from kabbalistic_core.poc import SeedService
from kabbalistic_core.poc_server import SeedRequestHandler


class _RecordingService:
    def __init__(self) -> None:
        self.csrf_token = "test-seed-csrf-token"
        self.run_payloads: list[dict] = []

    def health(self) -> dict:
        return {
            "service": "test-service",
            "status": "ready",
            "csrf_token": self.csrf_token,
        }

    def source_catalog(self) -> dict:
        return {"documents": []}

    def run_graph(self, payload: dict) -> dict:
        self.run_payloads.append(payload)
        return {"accepted": True}


class _QuietSeedRequestHandler(SeedRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class SeedServerSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _RecordingService()
        _QuietSeedRequestHandler.service = self.service
        self.port = 8765

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict, dict[str, str]]:
        handler = object.__new__(_QuietSeedRequestHandler)
        handler.service = self.service
        handler.server = SimpleNamespace(server_address=("127.0.0.1", self.port))
        handler.command = method
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.close_connection = True
        handler.rfile = BytesIO(body or b"")
        handler.wfile = BytesIO()
        from email.message import Message

        handler.headers = Message()
        for key, value in (headers or {}).items():
            handler.headers[key] = value
        if body is not None and "Content-Length" not in handler.headers:
            handler.headers["Content-Length"] = str(len(body))

        if method == "GET":
            handler.do_GET()
        elif method == "POST":
            handler.do_POST()
        else:
            raise AssertionError(f"Unsupported harness method: {method}")

        raw_headers, raw_body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(lines[0].split()[1])
        response_headers = {
            key.casefold(): value.strip()
            for key, value in (line.split(":", 1) for line in lines[1:] if ":" in line)
        }
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        return status, payload, response_headers

    def _valid_headers(self) -> dict[str, str]:
        return {
            "Host": f"127.0.0.1:{self.port}",
            "Origin": "http://127.0.0.1:3000",
            "Content-Type": "application/json; charset=utf-8",
            "X-SEED-CSRF": self.service.csrf_token,
        }

    def test_health_returns_the_startup_csrf_token_without_caching(self) -> None:
        status, payload, headers = self._request(
            "GET",
            "/api/health",
            headers={"Host": f"127.0.0.1:{self.port}"},
        )

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["csrf_token"], self.service.csrf_token)
        self.assertEqual(headers["cache-control"], "no-store")

    def test_real_service_generates_a_unique_token_and_returns_it_from_health(self) -> None:
        first = SeedService(adapter=_adapter(FakeLMStudioTransport(offline=True)))
        second = SeedService(adapter=_adapter(FakeLMStudioTransport(offline=True)))

        self.assertGreaterEqual(len(first.csrf_token), 32)
        self.assertNotEqual(first.csrf_token, second.csrf_token)
        self.assertEqual(first.health()["csrf_token"], first.csrf_token)

    def test_valid_local_json_post_requires_and_accepts_csrf_token(self) -> None:
        status, payload, _ = self._request(
            "POST",
            "/api/run",
            body=json.dumps({"intention": "bounded local request"}).encode("utf-8"),
            headers=self._valid_headers(),
        )

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload, {"accepted": True})
        self.assertEqual(
            self.service.run_payloads,
            [{"intention": "bounded local request"}],
        )

    def test_hostile_origin_is_rejected_before_service_dispatch(self) -> None:
        headers = self._valid_headers()
        headers["Origin"] = "https://attacker.example"

        status, _, _ = self._request(
            "POST",
            "/api/run",
            body=b'{"intention":"hostile origin"}',
            headers=headers,
        )

        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertEqual(self.service.run_payloads, [])

    def test_non_loopback_host_header_is_rejected_before_dispatch(self) -> None:
        headers = self._valid_headers()
        headers["Host"] = "attacker.example"

        status, _, _ = self._request(
            "POST",
            "/api/run",
            body=b'{"intention":"host header attack"}',
            headers=headers,
        )

        self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertEqual(self.service.run_payloads, [])

    def test_non_json_content_type_is_rejected_before_body_dispatch(self) -> None:
        headers = self._valid_headers()
        headers["Content-Type"] = "text/plain"

        status, _, _ = self._request(
            "POST",
            "/api/run",
            body=b'{"intention":"looks like json but is text"}',
            headers=headers,
        )

        self.assertEqual(status, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        self.assertEqual(self.service.run_payloads, [])

    def test_missing_or_wrong_csrf_token_is_rejected_before_dispatch(self) -> None:
        for token in (None, "wrong-token"):
            with self.subTest(token=token):
                headers = self._valid_headers()
                if token is None:
                    del headers["X-SEED-CSRF"]
                else:
                    headers["X-SEED-CSRF"] = token

                status, _, _ = self._request(
                    "POST",
                    "/api/run",
                    body=b'{"intention":"csrf attempt"}',
                    headers=headers,
                )

                self.assertEqual(status, HTTPStatus.FORBIDDEN)
        self.assertEqual(self.service.run_payloads, [])


if __name__ == "__main__":
    unittest.main()
