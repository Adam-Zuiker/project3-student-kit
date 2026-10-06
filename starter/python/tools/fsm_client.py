#!/usr/bin/env python3
"""fsm_client.py — Drive a running `make demo`'s finite_state_machine.

    python3 tools/fsm_client.py deploy fsm/fetch_wave.json   # validate + run a local file
    python3 tools/fsm_client.py deploy-saved fetch_wave      # load fsm/<id>.json through the node, run it
    python3 tools/fsm_client.py save my_dance fsm/draft.json # store a file through /fsm/save_to_file
    python3 tools/fsm_client.py list | get | start | stop
    python3 tools/fsm_client.py fire to_home
    python3 tools/fsm_client.py watch                        # print /fsm/status changes (Ctrl-C to stop)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_client import Client  # noqa: E402


def show(reply):
    print(json.dumps({k: reply[k] for k in ("result", "status", "values")}, indent=2))
    return 0 if reply["result"] else 1


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    c, cmd, rest = Client(), argv[0], argv[1:]
    if cmd == "deploy":
        with open(rest[0], encoding="utf-8") as f:
            return show(c.call("/fsm/update_logic", json.load(f)))
    if cmd == "deploy-saved":
        loaded = c.call("/fsm/load_from_file", {"identifier": rest[0]})
        return show(c.call("/fsm/update_logic", loaded["values"])) if loaded["result"] else show(loaded)
    if cmd == "save":
        with open(rest[1], encoding="utf-8") as f:
            return show(c.call("/fsm/save_to_file", dict(json.load(f), identifier=rest[0])))
    if cmd == "list":
        return show(c.call("/fsm/list_saved", {}))
    if cmd == "get":
        return show(c.call("/fsm/get_logic", {}))
    if cmd in ("start", "stop"):
        return show(c.call("/fsm/set_running", {"data": cmd == "start"}))
    if cmd == "fire":
        return show(c.call("/fsm/fire_transition", {"transition": rest[0]}))
    if cmd == "watch":
        c.send({"op": "subscribe", "topic": "/fsm/status"})
        last = None
        try:
            for line in c.stream:
                m = json.loads(line)
                if m.get("op") == "publish":
                    key = (m["msg"]["current_state"], m["msg"]["running"])
                    if key != last:
                        print(m["msg"])
                        last = key
        except KeyboardInterrupt:
            return 0
    print(f"unknown command {cmd!r}\n{__doc__}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
