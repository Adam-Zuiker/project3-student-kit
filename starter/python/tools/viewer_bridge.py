#!/usr/bin/env python3
"""viewer_bridge.py — Lets a browser watch the runtime. Not part of the
graded runtime; run it next to `make demo`:

    make demo                                   # terminal 1
    python3 tools/fk_client.py ../../robots/fetch/fetch.urdf    # load a robot
    python3 tools/viewer_bridge.py              # terminal 2, then open http://127.0.0.1:8080

Why a bridge: a browser cannot open a raw TCP socket, and the gateway speaks
newline-delimited JSON over TCP (not WebSocket). So this process is an
ordinary external client of 127.0.0.1:9095, like the autograder, and
re-serves what the page needs over plain HTTP:

    GET  /         tools/viewer/index.html
    GET  /events   Server-Sent Events: `robot` (link shapes + joint tree,
                   resent when a new description loads), `xform` (each
                   /xform_world), `fsm` (each /fsm/status), `gateway` (up/down)
    POST /call     {"service": ..., "args": ...} -> the service_response

The robot's shapes come from parsing robot_description with the runtime's
own fk.urdf, so the page never has to read XML. If the runtime restarts,
the bridge reconnects on its own.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from fk import urdf  # noqa: E402

INDEX_HTML = os.path.join(HERE, "viewer", "index.html")
TOPICS = ("/xform_world", "/fsm/status")
EVENT_FOR_TOPIC = {"/xform_world": "xform", "/fsm/status": "fsm"}


class Gateway:
    """One persistent client connection to the runtime, with reconnects."""

    def __init__(self, port: int, on_event) -> None:
        self._port = port
        self._on_event = on_event  # (event_name, data) -> None
        self._write_lock = threading.Lock()
        self._stream = None
        self._pending: Dict[str, Any] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count()
        self.connected = threading.Event()
        threading.Thread(target=self._run, name="gateway", daemon=True).start()

    def _send(self, message: Dict[str, Any]) -> None:
        with self._write_lock:
            if self._stream is None:
                raise ConnectionError("runtime not connected")
            self._stream.write((json.dumps(message) + "\n").encode())
            self._stream.flush()

    def call(self, service: str, args: Any, timeout: float = 5.0) -> Dict[str, Any]:
        call_id = f"viewer-{next(self._ids)}"
        slot = {"event": threading.Event(), "reply": None}
        with self._pending_lock:
            self._pending[call_id] = slot
        try:
            self._send({"op": "call_service", "service": service, "id": call_id, "args": args})
            if not slot["event"].wait(timeout):
                raise TimeoutError(f"{service} did not answer within {timeout} s")
            return slot["reply"]
        finally:
            with self._pending_lock:
                self._pending.pop(call_id, None)

    def _run(self) -> None:
        while True:
            try:
                sock = socket.create_connection(("127.0.0.1", self._port), timeout=2)
                sock.settimeout(None)
            except OSError:
                time.sleep(1.0)
                continue
            with self._write_lock:
                self._stream = sock.makefile("rwb")
            for topic in TOPICS:
                self._send({"op": "subscribe", "topic": topic})
            self.connected.set()
            self._on_event("gateway", {"connected": True})
            print(f"bridge: connected to runtime on 127.0.0.1:{self._port}", file=sys.stderr)
            try:
                for line in self._stream:
                    self._dispatch(json.loads(line))
            except (OSError, ValueError):
                pass
            self.connected.clear()
            with self._write_lock:
                self._stream = None
            sock.close()
            with self._pending_lock:
                for slot in self._pending.values():
                    slot["reply"] = {"result": False, "status": "runtime disconnected", "values": {}}
                    slot["event"].set()
            self._on_event("gateway", {"connected": False})
            print("bridge: runtime disconnected; retrying", file=sys.stderr)
            time.sleep(1.0)

    def _dispatch(self, m: Dict[str, Any]) -> None:
        op = m.get("op")
        if op == "service_response":
            with self._pending_lock:
                slot = self._pending.get(m.get("id"))
            if slot is not None:
                slot["reply"] = m
                slot["event"].set()
        elif op == "publish" and m.get("topic") in EVENT_FOR_TOPIC:
            self._on_event(EVENT_FOR_TOPIC[m["topic"]], m.get("msg"))


def robot_json(model: urdf.RobotModel, version: int) -> Dict[str, Any]:
    def visual(link: urdf.Link) -> Optional[Dict[str, Any]]:
        v = link.visual
        if v is None:
            return None
        return {"shape": v.shape, "params": v.params, "origin": {"xyz": v.origin.xyz, "rpy": v.origin.rpy}}

    return {
        "name": model.name,
        "version": version,
        "root": model.root,
        "links": [{"name": l.name, "visual": visual(l)} for l in model.links.values()],
        "joints": [{"name": j.name, "type": j.type, "parent": j.parent, "child": j.child} for j in model.joints],
    }


class Hub:
    """Fans events out to every open /events stream, and tracks the robot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: "set[queue.Queue]" = set()
        self.robot: Optional[Dict[str, Any]] = None

    def add(self) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=64)
        with self._lock:
            self._clients.add(q)
        return q

    def remove(self, q: "queue.Queue") -> None:
        with self._lock:
            self._clients.discard(q)

    def emit(self, event: str, data: Any) -> None:
        if event == "robot":
            self.robot = data
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait((event, data))
            except queue.Full:  # a stalled browser tab just misses frames
                pass


def watch_robot(gw: Gateway, hub: Hub) -> None:
    """Re-read the description whenever robot_state_publisher loads a new one."""
    loaded = None
    while True:
        time.sleep(0.5)
        if not gw.connected.is_set():
            continue
        try:
            status = gw.call("/param_server/get_param",
                             {"name": "/robot_state_publisher/description_status"})["values"]["value"]
            if not status or status.get("loaded_version") in (0, loaded):
                continue
            reply = gw.call("/param_server/get_param", {"name": "robot_description"})
            model = urdf.parse_robot_description(reply["values"]["value"])
        except Exception as exc:  # noqa: BLE001 - keep watching
            print(f"bridge: could not read robot: {exc}", file=sys.stderr)
            continue
        loaded = status["loaded_version"]
        hub.emit("robot", robot_json(model, loaded))
        print(f"bridge: robot {model.name!r} v{loaded}, {len(model.links)} links", file=sys.stderr)


def make_handler(gw: Gateway, hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, body: Any) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(INDEX_HTML, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/events":
                self._events()
            else:
                self.send_error(404)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = hub.add()
            try:
                self._write_event("gateway", {"connected": gw.connected.is_set()})
                if hub.robot is not None:
                    self._write_event("robot", hub.robot)
                while True:
                    try:
                        event, data = q.get(timeout=15)
                        self._write_event(event, data)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.remove(q)

        def _write_event(self, event: str, data: Any) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        def do_POST(self):
            if self.path != "/call":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                reply = gw.call(body["service"], body.get("args", {}))
                self._json(200, {k: reply.get(k) for k in ("result", "status", "values")})
            except Exception as exc:  # noqa: BLE001
                self._json(200, {"result": False, "status": str(exc), "values": {}})

    return Handler


def main() -> int:
    p = argparse.ArgumentParser(description="HTTP bridge between the runtime and tools/viewer/index.html")
    p.add_argument("--port", type=int, default=8080, help="HTTP port for the browser (default 8080)")
    p.add_argument("--runtime-port", type=int, default=9095)
    args = p.parse_args()

    hub = Hub()
    gw = Gateway(args.runtime_port, hub.emit)
    threading.Thread(target=watch_robot, args=(gw, hub), name="robot-watch", daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(gw, hub))
    server.daemon_threads = True
    print(f"bridge: open http://127.0.0.1:{args.port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
