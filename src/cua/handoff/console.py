"""OperatorConsole: the minimal local operator surface (Section 4).

Deliberately the smallest real thing that works: Python's stdlib
`http.server`, no FastAPI/uvicorn (nothing here needs routing, DI, or
async request handling -- it renders one page and handles one POST), no
frontend framework (one inline-styled HTML string), no auth, no queue, no
WebRTC/remote-desktop. The human does not interact with the *page this
serves* to operate the browser -- they interact with the already-visible
headed Playwright window directly, on the same machine; this console's
only job is to show *why* automation stopped (reason, step, expected vs.
observed state, a screenshot) and offer a Resume button. Production
considerations this deliberately skips (auth, TLS, multi-operator queuing,
a persistent dashboard) are listed in REPORT.md, not built here.

Runs its HTTP server on a background thread (stdlib's `http.server` is
synchronous) and hands `request_resume()` back to the owning asyncio event
loop via `call_soon_threadsafe` -- `asyncio.Event.set()` is not itself
thread-safe to call from a different thread, so this is the one place a
thread boundary is crossed, and it is crossed exactly once, for exactly
one method call.
"""
from __future__ import annotations

import asyncio
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cua.handoff.controller import HandoffController


class OperatorConsole:
    def __init__(self, controller: HandoffController, *, host: str = "127.0.0.1", port: int = 0):
        self._controller = controller
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("console is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        console = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass  # keep test/demo stdout quiet -- not a debugging aid worth the noise

            def do_GET(self) -> None:  # noqa: N802 -- http.server's naming convention
                if self.path == "/screenshot":
                    console._serve_screenshot(self)
                else:
                    console._serve_index(self)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)  # discard -- the form carries no fields we read
                if self.path == "/resume":
                    console._handle_resume(self)
                else:
                    self.send_response(404)
                    self.end_headers()

        self._server = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _serve_index(self, handler: BaseHTTPRequestHandler) -> None:
        body = self._render_page().encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _serve_screenshot(self, handler: BaseHTTPRequestHandler) -> None:
        record = self._controller.current_intervention
        ref = record.request.evidence if record is not None else None
        if ref is None or not Path(ref.path).exists():
            handler.send_response(404)
            handler.end_headers()
            return
        data = Path(ref.path).read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "image/png")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def _handle_resume(self, handler: BaseHTTPRequestHandler) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._controller.request_resume)
        handler.send_response(303)
        handler.send_header("Location", "/")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _render_page(self) -> str:
        state = html.escape(self._controller.state.value)
        record = self._controller.current_intervention
        if record is None:
            body = f"<p>No open intervention. Control owner: <b>{state}</b></p>"
        else:
            req = record.request

            def esc(value: str | None) -> str:
                return html.escape(value) if value else "-"

            screenshot = '<img src="/screenshot" alt="evidence screenshot" style="max-width:640px;border:1px solid #999">' if req.evidence else "<p><i>(no screenshot captured)</i></p>"
            body = f"""
              <h1>Intervention required</h1>
              <p><b>Control owner:</b> {state}</p>
              <p><b>Run:</b> {esc(req.run_id)} (mode={esc(req.mode)})</p>
              <p><b>Capability:</b> {esc(req.capability_id)} {esc(req.capability_version)}</p>
              <p><b>Goal:</b> {esc(req.goal)}</p>
              <p><b>Step:</b> {esc(req.step_id)}</p>
              <p><b>Reason:</b> {esc(req.reason_code)} -- {esc(req.reason_message)}</p>
              <p><b>Expected state:</b> {esc(req.expected_state)}</p>
              <p><b>Observed state:</b> {esc(req.observed_state)}</p>
              <p><b>Current URL:</b> {esc(req.current_url)}</p>
              {screenshot}
              <form method="POST" action="/resume"><button type="submit">Resume</button></form>
            """
        return f"<!doctype html><html><head><title>Operator console</title></head><body>{body}</body></html>"
