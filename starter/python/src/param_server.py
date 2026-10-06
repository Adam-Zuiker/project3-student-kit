"""param_server.py — `/param_server/set_param` and `/param_server/get_param`
(spec/FK_API.md, "param_server").

A dict of name -> (value, version) behind one lock. The server never
interprets a value: a broken URDF is stored like anything else, and
judging it is robot_state_publisher's job.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from broker import Broker
from services.provider import InProcessProvider, register as register_provider


class ParamStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._params: Dict[str, Tuple[Any, int]] = {}

    def set(self, name: str, value: Any) -> int:
        with self._lock:
            _, version = self._params.get(name, (None, 0))
            version += 1
            self._params[name] = (value, version)
            return version

    def get(self, name: str) -> Tuple[Any, int, bool]:
        """Returns `(value, version, found)`."""
        with self._lock:
            if name not in self._params:
                return None, 0, False
            value, version = self._params[name]
            return value, version, True


def _name_of(args: Any) -> Any:
    return args.get("name") if isinstance(args, dict) else None


def register(broker: Broker) -> Tuple[List[InProcessProvider], ParamStore]:
    store = ParamStore()

    def set_param(args: Any) -> Tuple[Any, bool, str]:
        name = _name_of(args)
        if not isinstance(name, str):
            return {}, False, "args.name is required and must be a string"
        return {"version": store.set(name, args.get("value"))}, True, ""

    def get_param(args: Any) -> Tuple[Any, bool, str]:
        name = _name_of(args)
        if not isinstance(name, str):
            return {"value": None, "version": 0}, False, "args.name is required and must be a string"
        value, version, found = store.get(name)
        if not found:
            return {"value": None, "version": 0}, False, f"parameter {name!r} is not set"
        return {"value": value, "version": version}, True, ""

    providers = [
        register_provider(broker, "/param_server/set_param", set_param),
        register_provider(broker, "/param_server/get_param", get_param),
    ]
    return providers, store
