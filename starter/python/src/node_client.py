"""node_client.py — The pieces an in-process *node* needs to talk to the
broker the same way an external client does, without a socket.

The broker only knows `MessageSink`s (an `id` plus `send(message)`, see
broker.py). Project 2 already used that trick for service *providers*
(services/provider.py) and topic subscribers (arm_sim_node's
`_JointTrajectorySink`). Project 3's nodes also need to be service
*callers* — robot_state_publisher reads `robot_description` through
`/param_server/get_param` — so this module adds:

  - `ServiceClient`: a sink that turns the broker's asynchronous
    call_service / service_response round trip into a blocking `call()`.
  - `TopicSubscriber`: a sink that hands each published `msg` to a callback.
  - `PeriodicLoop`: one daemon thread running `fn()` at a fixed rate,
    the shape every publisher node here takes.
"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from broker import Broker

log = logging.getLogger("node_client")


class _Slot:
    """One outstanding call's mailbox: set once by the broker's thread,
    waited on by the caller's."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: Optional[Dict[str, Any]] = None


class ServiceClient:
    """Blocking service calls through the broker, from inside the process."""

    def __init__(self, broker: Broker, name: str) -> None:
        self.id = f"client:{name}"
        self._broker = broker
        self._lock = threading.Lock()
        self._pending: Dict[str, _Slot] = {}
        self._ids = itertools.count(1)

    def send(self, message: Dict[str, Any]) -> None:
        if message.get("op") != "service_response":
            return
        with self._lock:
            slot = self._pending.pop(message.get("id"), None)
        if slot is not None:
            slot.message = message
            slot.event.set()

    def call(self, service: str, args: Any, timeout: float = 5.0) -> Tuple[Any, bool, str]:
        """Returns `(values, result, status)`; a timeout reads as a failed call."""
        call_id = f"{self.id}-{next(self._ids)}"
        slot = _Slot()
        # Register before calling: with no provider, the broker answers
        # synchronously, inside call_service itself.
        with self._lock:
            self._pending[call_id] = slot
        self._broker.call_service(self, call_id, service, args)
        if not slot.event.wait(timeout):
            with self._lock:
                self._pending.pop(call_id, None)
            return {}, False, "timeout"
        assert slot.message is not None
        message = slot.message
        return message.get("values"), bool(message.get("result")), message.get("status", "")


class TopicSubscriber:
    """Calls `callback(msg)` for every publish on `topic`.

    The callback runs on the *publisher's* thread (often a client's
    connection reader), so it should only stash data under a lock and
    return; the heavy work belongs on the node's own `PeriodicLoop`.
    """

    def __init__(self, broker: Broker, topic: str, callback: Callable[[Any], None]) -> None:
        self.id = f"sub:{topic}"
        self._topic = topic
        self._callback = callback
        broker.subscribe(self, topic)

    def send(self, message: Dict[str, Any]) -> None:
        if message.get("op") != "publish" or message.get("topic") != self._topic:
            return
        try:
            self._callback(message.get("msg"))
        except Exception:  # noqa: BLE001 - a bad message must not kill the publisher's thread
            log.exception("%s callback failed", self.id)


class PeriodicLoop:
    """Runs `fn()` every `period` seconds on a daemon thread until stopped.
    An exception in one tick is logged, and the loop keeps going."""

    def __init__(self, name: str, period: float, fn: Callable[[], None]) -> None:
        self.id = name
        self._period = period
        self._fn = fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> "PeriodicLoop":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._fn()
            except Exception:  # noqa: BLE001
                log.exception("%s tick failed", self.id)
            self._stop.wait(self._period)
