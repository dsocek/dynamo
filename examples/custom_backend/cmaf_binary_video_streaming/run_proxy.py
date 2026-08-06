# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import mimetypes
import sys
import threading
import time
import uuid
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
CLIENT_HTML = EXAMPLE_DIR / "client.html"
VIDEO_ROUTE_PREFIX = "/v1/videos"
CF_SESSION_PREFIX = "cf_session="
CF_CLOSE_ANNOTATION = "cf_close"
DEFAULT_GATE_IDLE_TIMEOUT = 180.0
# A single scene is seconds, but a stalled one must not pin the gate forever.
# Bytes flowing refresh this, so it only trips on a genuinely dead stream.
DEFAULT_GATE_STUCK_TIMEOUT = 120.0
# Upstream socket ceiling. The old hardcoded 300 s is why an XPU-OOM'd stream
# left a worker thread parked long enough to pin the gate.
DEFAULT_UPSTREAM_TIMEOUT = 180.0
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class GpuGate:
    """One generation at a time, held across a whole cf_session.

    The pipeline is a single DiT worker: two overlapping rollouts contend on the
    shared-memory broadcast block and can wedge it (observed 2026-08-05 --
    a second session opened mid-denoise, the loop froze at 18/101 steps, and
    every later request failed with a 60s acquire_write timeout until restart).

    A per-request flag is not enough. A storyboard is N chained POSTs, so the
    gate must stay held between scenes -- that gap is exactly where the second
    caller got in. Ownership is therefore keyed by cf_session id: the scene that
    opens a session takes the gate, later scenes on that id pass straight
    through, and cf_close (or an idle timeout, or a dropped connection on the
    last in-flight scene) releases it.
    """

    def __init__(
        self,
        idle_timeout: float = DEFAULT_GATE_IDLE_TIMEOUT,
        stuck_timeout: float = DEFAULT_GATE_STUCK_TIMEOUT,
    ) -> None:
        self._lock = threading.Lock()
        self._idle_timeout = idle_timeout
        self._stuck_timeout = stuck_timeout
        self._owner: str | None = None
        self._in_flight = 0
        self._touched = 0.0

    def _expired_locked(self, now: float) -> bool:
        """Two ways an owner loses the gate without ever sending cf_close.

        `idle` covers the ordinary abandoned tab: nothing in flight, no next
        scene. `stuck` is the one that bit us on 2026-08-05 -- the DiT card hit
        XPU-OOM mid-scene, the viewer walked away, and the proxy worker thread
        stayed parked on the dead socket. in_flight never fell to 0, so an
        idle-only check could never fire and the gate was pinned until restart.
        """
        if self._owner is None:
            return False
        age = now - self._touched
        if self._in_flight == 0:
            return age > self._idle_timeout
        return age > self._stuck_timeout

    def try_acquire(self, session: str | None) -> tuple[bool, str | None, str]:
        """Return (allowed, owner_token, reason). Non-blocking, so a rejected
        caller gets an immediate 429 instead of a socket that hangs for minutes."""
        token = session or f"anon-{uuid.uuid4().hex[:12]}"
        now = time.monotonic()
        with self._lock:
            if self._expired_locked(now):
                why = "stuck" if self._in_flight else "idle"
                limit = self._stuck_timeout if self._in_flight else self._idle_timeout
                sys.stderr.write(
                    f"[demo-proxy] gate: released {why} session {self._owner} "
                    f"after {limit:.0f}s (in_flight={self._in_flight})\n"
                )
                self._owner = None
                self._in_flight = 0

            if self._owner is None:
                self._owner = token
                self._in_flight = 1
                self._touched = now
                return True, token, "acquired"

            if self._owner == token:
                self._in_flight += 1
                self._touched = now
                return True, token, "same-session"

            return False, None, f"busy with session {self._owner}"

    def heartbeat(self, token: str) -> None:
        """Bytes are still flowing, so this owner is alive, not stuck."""
        with self._lock:
            if self._owner == token:
                self._touched = time.monotonic()

    def finish(self, token: str, release: bool) -> None:
        with self._lock:
            if self._owner != token:
                return
            self._in_flight = max(0, self._in_flight - 1)
            self._touched = time.monotonic()
            if release and self._in_flight == 0:
                self._owner = None

    def status(self) -> dict:
        with self._lock:
            busy = self._owner is not None and not self._expired_locked(
                time.monotonic()
            )
            return {
                "busy": busy,
                "session": self._owner if busy else None,
                "in_flight": self._in_flight,
            }


class DemoProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        frontend_host: str,
        frontend_port: int,
        gate_idle_timeout: float = DEFAULT_GATE_IDLE_TIMEOUT,
        gate_stuck_timeout: float = DEFAULT_GATE_STUCK_TIMEOUT,
        upstream_timeout: float = DEFAULT_UPSTREAM_TIMEOUT,
    ) -> None:
        super().__init__(server_address, DemoProxyHandler)
        self.frontend_host = frontend_host
        self.frontend_port = frontend_port
        self.static_root = EXAMPLE_DIR
        self.upstream_timeout = upstream_timeout
        self.gpu_gate = GpuGate(
            idle_timeout=gate_idle_timeout,
            stuck_timeout=gate_stuck_timeout,
        )


class DemoProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "cmaf-binary-demo/0.1"

    def do_GET(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle(head_only=True)

    def do_POST(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def _handle(self, head_only: bool = False) -> None:
        if self.path == "/" or self.path == "/client.html":
            self._serve_static(CLIENT_HTML, head_only=head_only)
            return

        if self.path == "/demo/gpu-status":
            self._send_json(200, self.server.gpu_gate.status())
            return

        if self.path.startswith("/v1/") or self.path in {
            "/live",
            "/ready",
            "/health",
            "/metrics",
        }:
            self._proxy_request(head_only=head_only)
            return

        candidate = self._safe_static_path(self.path)
        if candidate and candidate.is_file():
            self._serve_static(candidate, head_only=head_only)
            return

        self.send_error(404, "Not found")

    def _safe_static_path(self, raw_path: str) -> Path | None:
        cleaned = raw_path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if not cleaned:
            return None
        candidate = (self.server.static_root / cleaned).resolve()
        try:
            candidate.relative_to(self.server.static_root)
        except ValueError:
            return None
        return candidate

    def _serve_static(self, path: Path, head_only: bool = False) -> None:
        body = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        # The demo page changes often during a bring-up; a cached copy silently
        # hides edits (and a corporate proxy will happily serve its own copy).
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _inspect_generation_body(body: bytes | None) -> tuple[str | None, bool]:
        """Pull (cf_session, is_closing) out of a generation request body.

        Unparseable bodies yield (None, True): the request still gets gated, but
        as a one-shot that releases on completion.
        """
        if not body:
            return None, True
        try:
            annotations = (
                json.loads(body).get("nvext", {}).get("annotations", []) or []
            )
        except (ValueError, AttributeError):
            return None, True
        strings = [i for i in annotations if isinstance(i, str)]
        session = None
        for item in strings:
            if item.startswith(CF_SESSION_PREFIX):
                session = item[len(CF_SESSION_PREFIX) :].strip() or None
        if session is None:
            # Single-prompt request: one shot, release as soon as it finishes.
            return None, True
        # A session-scoped scene keeps the gate until the scene carrying cf_close.
        return session, CF_CLOSE_ANNOTATION in strings

    def _proxy_request(self, head_only: bool = False) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None

        # Gate only actual generation work; /v1/models and the health routes
        # must keep answering while a render is in flight.
        gated = self.command == "POST" and self.path.startswith(VIDEO_ROUTE_PREFIX)
        token: str | None = None
        release = True
        if gated:
            session, release = self._inspect_generation_body(body)
            allowed, token, reason = self.server.gpu_gate.try_acquire(session)
            if not allowed:
                sys.stderr.write(f"[demo-proxy] gate: rejected 429 -- {reason}\n")
                self._send_json(
                    429,
                    {
                        "error": {
                            "message": (
                                "The demo GPU is already rendering another "
                                "request. Wait for it to finish and try again."
                            ),
                            "type": "gpu_busy",
                            "code": "gpu_busy",
                        }
                    },
                )
                return

        try:
            status = self._forward(self.path, body, head_only=head_only, token=token)
            # A scene that FAILED must not keep holding the session's gate.
            # release is False for session-scoped scenes (the gate is meant to
            # span the whole storyboard), but that reasoning only holds while
            # scenes are succeeding: the next scene is what would have released
            # it, and after an error there may be no next scene. Holding on
            # would pin the card until the idle timeout -- observed as repeated
            # "429 busy with session <id>" minutes after a scene 404'd with a
            # stale model id, with the GPUs sitting at 0% the whole time.
            if status is not None and status >= 400:
                sys.stderr.write(
                    f"[demo-proxy] gate: upstream {status}, releasing session early\n"
                )
                release = True
        except (TimeoutError, OSError) as exc:
            # A dead upstream must not leave the caller's socket parked: that is
            # what pinned the gate on 2026-08-05. Release happens in `finally`.
            sys.stderr.write(f"[demo-proxy] upstream failed: {exc!r}\n")
            release = True
        finally:
            if token is not None:
                self.server.gpu_gate.finish(token, release)

    def _forward(
        self,
        path: str,
        body: bytes | None,
        head_only: bool = False,
        token: str | None = None,
    ) -> int | None:
        """Returns the upstream HTTP status, or None if it never answered.

        The caller needs the status to decide whether to release the gate early,
        so a failure that is a completed round-trip (4xx/5xx) stays
        distinguishable from a healthy stream.
        """
        status: int | None = None
        upstream_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        upstream_headers[
            "Host"
        ] = f"{self.server.frontend_host}:{self.server.frontend_port}"

        conn = http.client.HTTPConnection(
            self.server.frontend_host,
            self.server.frontend_port,
            timeout=self.server.upstream_timeout,
        )
        try:
            conn.request(self.command, path, body=body, headers=upstream_headers)
            response = conn.getresponse()
            status = response.status
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            self.end_headers()

            if head_only:
                response.read()
                return status

            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if token is not None:
                    # Progress means alive: keeps a long but healthy render from
                    # tripping the stuck-owner timeout.
                    self.server.gpu_gate.heartbeat(token)
        except (BrokenPipeError, ConnectionResetError):
            # The caller hung up mid-stream. Treat it as a failure so the gate is
            # released now rather than waiting on the stuck-owner timeout: a
            # browser that navigates away mid-storyboard sends no further scene,
            # so nothing else would release it.
            status = status if status is not None and status >= 400 else 499
        finally:
            conn.close()
        return status

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[demo-proxy] {self.address_string()} - {fmt % args}\n")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the CMAF binary demo page and proxy an already-running "
        "Dynamo frontend under one browser origin."
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Browser-facing bind address for the demo proxy (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=8080,
        help="Browser-facing port for the demo proxy (default: 8080)",
    )
    parser.add_argument(
        "--frontend-host",
        default="127.0.0.1",
        help="Host where the running Dynamo frontend listens (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=18001,
        help="Port of the running Dynamo frontend to proxy (default: 18001)",
    )
    parser.add_argument(
        "--frontend-timeout-seconds",
        type=float,
        default=30.0,
        help="How long to wait for the frontend to respond before serving (default: 30)",
    )
    parser.add_argument(
        "--gate-idle-timeout-seconds",
        type=float,
        default=DEFAULT_GATE_IDLE_TIMEOUT,
        help="Release the one-at-a-time GPU gate if a session goes this long "
        "between scenes, e.g. an abandoned browser tab that never sent cf_close "
        f"(default: {DEFAULT_GATE_IDLE_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--gate-stuck-timeout-seconds",
        type=float,
        default=DEFAULT_GATE_STUCK_TIMEOUT,
        help="Release the gate if the owner has a request in flight but no bytes "
        "have moved for this long, e.g. a render that died on the GPU while the "
        f"viewer walked away (default: {DEFAULT_GATE_STUCK_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--upstream-timeout-seconds",
        type=float,
        default=DEFAULT_UPSTREAM_TIMEOUT,
        help="Socket timeout when talking to the frontend "
        f"(default: {DEFAULT_UPSTREAM_TIMEOUT:.0f})",
    )
    return parser


def wait_for_frontend(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1)
            conn.request("GET", "/live")
            response = conn.getresponse()
            response.read()
            conn.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(
        f"Timed out waiting for frontend on http://{host}:{port} ({last_error})"
    )


def main() -> int:
    args = create_parser().parse_args()

    wait_for_frontend(
        args.frontend_host,
        args.frontend_port,
        args.frontend_timeout_seconds,
    )

    server = DemoProxyServer(
        (args.bind, args.proxy_port),
        frontend_host=args.frontend_host,
        frontend_port=args.frontend_port,
        gate_idle_timeout=args.gate_idle_timeout_seconds,
        gate_stuck_timeout=args.gate_stuck_timeout_seconds,
        upstream_timeout=args.upstream_timeout_seconds,
    )

    print()
    print("CMAF binary demo proxy is ready")
    print(f"  Browser URL: http://{args.bind}:{args.proxy_port}/")
    print(f"  Proxied frontend: http://{args.frontend_host}:{args.frontend_port}/")
    print("  The browser now sees one origin, so no Dynamo CORS change is needed.")
    print(
        f"  One generation at a time; extra callers get HTTP 429 "
        f"(gate frees after {args.gate_idle_timeout_seconds:.0f}s idle)."
    )
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
