"""Loopback-only JSON server for the guided local SEED interface."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import secrets
from typing import Any
from urllib.parse import urlparse

from .neural import ModelProtocolError
from .poc import SeedService


_ALLOWED_ORIGINS = frozenset(
    {
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    }
)


class SeedRequestHandler(BaseHTTPRequestHandler):
    service: SeedService
    server_version = "SEEDLocal/0.1"

    def log_message(self, format: str, *args: object) -> None:
        print(f"SEED API: {format % args}")

    def _origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin in _ALLOWED_ORIGINS else None

    def _host_is_allowed(self) -> bool:
        raw_host = self.headers.get("Host", "")
        try:
            parsed = urlparse(f"//{raw_host}")
            port = parsed.port
        except ValueError:
            return False
        expected_port = int(self.server.server_address[1])
        return (
            parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.username is None
            and parsed.password is None
            and port == expected_port
        )

    def _origin_is_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in _ALLOWED_ORIGINS

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _read_payload(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
        if content_type != "application/json":
            raise TypeError("State-changing routes require Content-Type: application/json.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer.") from exc
        if length <= 0 or length > 65536:
            raise ValueError("Request body must be between 1 byte and 64 KiB.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be a UTF-8 JSON object.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._host_is_allowed() or not self._origin_is_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "Local request boundary rejected."})
            return
        self.send_response(HTTPStatus.NO_CONTENT.value)
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-SEED-CSRF")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_is_allowed() or not self._origin_is_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "Local request boundary rejected."})
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(HTTPStatus.OK, self.service.health())
            return
        if path == "/api/sources":
            self._send(HTTPStatus.OK, self.service.source_catalog())
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "Unknown local API route."})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._host_is_allowed() or not self._origin_is_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "Local request boundary rejected."})
            return
        supplied_token = self.headers.get("X-SEED-CSRF", "")
        if not secrets.compare_digest(supplied_token, self.service.csrf_token):
            self._send(HTTPStatus.FORBIDDEN, {"error": "Missing or invalid local request token."})
            return
        try:
            payload = self._read_payload()
            if path == "/api/run":
                result = self.service.run_graph(payload)
            elif path == "/api/control":
                result = self.service.run_control(str(payload.get("session_id", "")))
            elif path == "/api/feedback":
                result = self.service.receive_feedback(
                    str(payload.get("session_id", "")),
                    str(payload.get("feedback", "")),
                )
            elif path == "/api/continue":
                result = self.service.continue_with_feedback(
                    str(payload.get("session_id", "")),
                    str(payload.get("feedback", "")),
                )
            elif path == "/api/export":
                result = self.service.export_session(str(payload.get("session_id", "")))
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "Unknown local API route."})
                return
        except KeyError as exc:
            self._send(HTTPStatus.NOT_FOUND, {"error": str(exc).strip("'")})
            return
        except ValueError as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except TypeError as exc:
            self._send(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": str(exc)})
            return
        except ModelProtocolError as exc:
            self._send(
                HTTPStatus.BAD_GATEWAY,
                {
                    "error": str(exc),
                    "kind": "model_protocol_error",
                    "note": "No invalid model contribution was accepted into graph state.",
                },
            )
            return
        except Exception as exc:  # pragma: no cover - final server boundary
            print(f"SEED API internal error: {type(exc).__name__}: {exc}")
            self._send(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "The local cycle failed before a result was sealed."},
            )
            return
        self._send(HTTPStatus.OK, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the loopback-only SEED POC backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The POC server may bind only to loopback.")
    SeedRequestHandler.service = SeedService(distribution_mode="public")
    server = HTTPServer((args.host, args.port), SeedRequestHandler)
    print(f"SEED backend ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
