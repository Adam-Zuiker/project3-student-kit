"""finite_state_machine.py — Runs a choreography: a table of joint-space
setpoints with named transitions (spec/FK_API.md, "finite_state_machine").
Started by `make demo` only.

On entering a state it publishes that state's target on `/joint_trajectory`
(joint_state_publisher servos toward it). A 10 Hz loop watches the latest
`/joint_states`, and when `fsm_logic.arrived(...)` says every target joint
is within `epsilon`, follows the state's `on_arrival` transition. The same
loop publishes `/fsm/status`.

Locking: the services, the loop, and the /joint_states callback all share
`self._lock`, but nothing is *published* while holding it. `_enter` returns
the message to send, and callers publish it after releasing the lock.

Saved choreographies live in `<project root>/fsm/<identifier>.json`, and an
identifier must match `[A-Za-z0-9_-]{1,64}`, so a request can never name a
path outside that directory (no `/`, no `..`).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from broker import Broker
from fsm_logic import arrived, validate_fsm
from node_client import PeriodicLoop, TopicSubscriber
from services.provider import register as register_provider

log = logging.getLogger("finite_state_machine")

TICK_PERIOD_S = 0.1
FSM_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fsm"))
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

Reply = Tuple[Any, bool, str]


def _identifier(args: Any) -> Optional[str]:
    ident = args.get("identifier") if isinstance(args, dict) else None
    if isinstance(ident, str) and _IDENTIFIER_RE.fullmatch(ident):
        return ident
    return None


def _trajectory(state: Dict[str, Any]) -> Dict[str, Any]:
    target = state["target"]
    return {
        "joint_names": list(target["joint_names"]),
        "points": [{"positions": [float(p) for p in target["positions"]],
                    "velocities": [], "accelerations": [],
                    "time_from_start": {"sec": 0, "nanosec": 0}}],
    }


class FiniteStateMachine:
    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._lock = threading.Lock()
        self._table: Optional[Dict[str, Any]] = None
        self._states: Dict[str, Dict[str, Any]] = {}
        self._current: Optional[str] = None
        self._entered_at = time.monotonic()
        self._arrived_at: Optional[float] = None
        self._last_transition: Optional[str] = None
        self._running = False
        self._positions: Dict[str, float] = {}

        TopicSubscriber(broker, "/joint_states", self._on_joint_states)
        services = {
            "/fsm/update_logic": self._update_logic,
            "/fsm/get_logic": self._get_logic,
            "/fsm/set_running": self._set_running,
            "/fsm/fire_transition": self._fire_transition,
            "/fsm/save_to_file": self._save_to_file,
            "/fsm/load_from_file": self._load_from_file,
            "/fsm/list_saved": self._list_saved,
        }
        self._providers = [register_provider(broker, name, fn) for name, fn in services.items()]
        self._loop = PeriodicLoop("finite_state_machine", TICK_PERIOD_S, self._tick)

    def start(self) -> List[Any]:
        return self._providers + [self._loop.start()]

    # -- state changes (call with self._lock held) ------------------------

    def _enter(self, name: str, via: Optional[str]) -> Dict[str, Any]:
        """Switch to state `name`; returns the /joint_trajectory to publish
        once the lock is released."""
        self._current = name
        self._entered_at = time.monotonic()
        self._arrived_at = None
        if via is not None:
            self._last_transition = via
        log.info("-> %s%s", name, f" (via {via})" if via else "")
        return _trajectory(self._states[name])

    def _take(self, transition: str) -> Optional[Dict[str, Any]]:
        """Follow `transition` out of the current state, if it has one."""
        state = self._states[self._current]
        for t in state["transitions"]:
            if t["name"] == transition:
                return self._enter(t["next_state"], transition)
        return None

    def _publish(self, trajectory: Optional[Dict[str, Any]]) -> None:
        if trajectory is not None:
            self._broker.publish("/joint_trajectory", trajectory)

    # -- inputs -----------------------------------------------------------

    def _on_joint_states(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        positions = {
            n: float(p) for n, p in zip(msg.get("name") or [], msg.get("position") or [])
            if isinstance(n, str) and isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p)
        }
        with self._lock:
            self._positions = positions

    def _tick(self) -> None:
        trajectory = None
        with self._lock:
            if self._running and self._current is not None:
                state = self._states[self._current]
                transition = state.get("on_arrival")
                now = time.monotonic()
                if not arrived(state, self._positions):
                    self._arrived_at = None
                elif self._arrived_at is None:
                    self._arrived_at = now  # start the optional "hold" (seconds) once there
                if transition and self._arrived_at is not None and now - self._arrived_at >= state.get("hold", 0):
                    trajectory = self._take(transition)
            status = {
                "current_state": self._current,
                "time_in_state": round(time.monotonic() - self._entered_at, 3) if self._current else 0.0,
                "last_transition": self._last_transition,
                "running": self._running,
            }
        self._publish(trajectory)
        self._broker.publish("/fsm/status", status)

    # -- services -----------------------------------------------------------

    def _update_logic(self, args: Any) -> Reply:
        error = validate_fsm(args)
        if error is not None:
            return {}, False, f"invalid FSM: {error}"
        table = {"initial_state": args["initial_state"], "states": copy.deepcopy(args["states"])}
        with self._lock:
            self._table = table
            self._states = {s["name"]: s for s in table["states"]}
            self._last_transition = None
            self._running = True
            trajectory = self._enter(table["initial_state"], None)
            current = self._current
        self._publish(trajectory)
        return {"current_state": current}, True, ""

    def _get_logic(self, args: Any) -> Reply:
        with self._lock:
            if self._table is None:
                return {"initial_state": None, "states": []}, False, "no FSM loaded"
            return copy.deepcopy(self._table), True, ""

    def _set_running(self, args: Any) -> Reply:
        data = args.get("data") if isinstance(args, dict) else None
        if data is not None and not isinstance(data, bool):
            return {}, False, "args.data must be a boolean"
        trajectory = None
        with self._lock:
            if data is not None:
                resuming = data and not self._running and self._current is not None
                self._running = data
                if resuming:  # re-send the setpoint in case something moved the joints meanwhile
                    trajectory = _trajectory(self._states[self._current])
            running = self._running
        self._publish(trajectory)
        return {"running": running}, True, ""

    def _fire_transition(self, args: Any) -> Reply:
        name = args.get("transition") if isinstance(args, dict) else None
        if not isinstance(name, str):
            return {}, False, "args.transition must be a string"
        with self._lock:
            if not self._running or self._current is None:
                return {}, False, "FSM is not running"
            trajectory = self._take(name)
            current = self._current
        if trajectory is None:
            return {}, False, f"state {current!r} has no transition {name!r}"
        self._publish(trajectory)
        return {"current_state": current}, True, ""

    def _save_to_file(self, args: Any) -> Reply:
        ident = _identifier(args)
        if ident is None:
            return {}, False, "identifier must match [A-Za-z0-9_-]{1,64}"
        doc = {"initial_state": args.get("initial_state"), "states": args.get("states")}
        os.makedirs(FSM_DIR, exist_ok=True)
        path = os.path.join(FSM_DIR, f"{ident}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        return {"path": os.path.join("fsm", f"{ident}.json")}, True, ""

    def _load_from_file(self, args: Any) -> Reply:
        ident = _identifier(args)
        if ident is None:
            return {}, False, "identifier must match [A-Za-z0-9_-]{1,64}"
        try:
            with open(os.path.join(FSM_DIR, f"{ident}.json"), encoding="utf-8") as f:
                doc = json.load(f)
        except FileNotFoundError:
            return {}, False, f"no saved FSM {ident!r}"
        except (OSError, json.JSONDecodeError) as exc:
            return {}, False, f"could not read {ident!r}: {exc}"
        if not isinstance(doc, dict):
            return {}, False, f"saved FSM {ident!r} is not an object"
        return {"initial_state": doc.get("initial_state"), "states": doc.get("states")}, True, ""

    def _list_saved(self, args: Any) -> Reply:
        try:
            names = os.listdir(FSM_DIR)
        except FileNotFoundError:
            names = []
        idents = sorted(n[:-5] for n in names if n.endswith(".json") and _IDENTIFIER_RE.fullmatch(n[:-5]))
        return {"identifiers": idents}, True, ""


def register(broker: Broker) -> List[Any]:
    return FiniteStateMachine(broker).start()
