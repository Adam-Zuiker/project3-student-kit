"""main.py — Entry point for the Project 3 FK runtime. `make run` executes
this: gateway on 127.0.0.1:9095 plus param_server, robot_state_publisher,
and robot_world_state_publisher, all in one process on one shared Broker.

Structurally identical to Projects 1 and 2's main.py; only the set of
registered nodes differs. `--demo` (used by `make demo` only) will add the
ungraded joint_state_publisher and finite_state_machine.
"""

from __future__ import annotations

import logging
import signal
import sys

import finite_state_machine
import joint_state_publisher
import logging_setup
import param_server
import robot_state_publisher
import robot_world_state_publisher
from broker import Broker
from server import TCPServer

log = logging.getLogger("main")


def main(argv: list[str]) -> int:
    logging_setup.configure()
    demo = "--demo" in argv

    broker = Broker()
    stoppables: list = []
    providers, _store = param_server.register(broker)
    stoppables += providers
    stoppables += robot_state_publisher.register(broker)
    stoppables += robot_world_state_publisher.register(broker)
    if demo:
        stoppables += joint_state_publisher.register(broker)
        stoppables += finite_state_machine.register(broker)
    log.info("registered: %s", ", ".join(s.id for s in stoppables))

    server = TCPServer(broker)
    try:
        server.bind_and_listen()
    except OSError as exc:
        log.error("could not bind 127.0.0.1:9095: %s", exc)
        return 1

    def handle_signal(signum: int, frame) -> None:  # noqa: ANN001 - stdlib signature
        log.info("received signal %s, shutting down", signal.Signals(signum).name)
        server.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    log.info("Project 3 runtime ready on 127.0.0.1:9095")
    try:
        server.serve_forever()
    finally:
        for stoppable in stoppables:
            stoppable.stop()
    log.info("shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
