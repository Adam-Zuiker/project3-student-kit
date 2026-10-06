"""robot_state_publisher.py — `robot_description` + `/joint_states` in,
`/tf` out (spec/FK_API.md, "robot_state_publisher").

Everything that touches "which robot is loaded" runs on ONE thread, the
10 Hz loop: poll the description, maybe swap robots and write the status,
then publish `/tf`. Because the swap, the status write, and every `/tf`
publish are sequential on that thread, "every /tf published after the
status reflects the new robot" holds by construction, with no lock dance.
The only other thread that touches this node is the `/joint_states`
callback, which just replaces `(positions, stamp)` under a lock.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional

from broker import Broker
from fk import transforms, urdf
from fk.kinematics import joint_transform
from node_client import PeriodicLoop, ServiceClient, TopicSubscriber

log = logging.getLogger("robot_state_publisher")

PUBLISH_PERIOD_S = 0.1  # 10 Hz, like the reference
STATUS_PARAM = "/robot_state_publisher/description_status"


def _zero_stamp() -> Dict[str, int]:
    return {"sec": 0, "nanosec": 0}


class RobotStatePublisher:
    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._client = ServiceClient(broker, "robot_state_publisher")

        # Loop-thread-only state.
        self._robot: Optional[urdf.RobotModel] = None
        self._loaded_version = 0
        self._processed_version = 0

        # Shared with the /joint_states callback.
        self._lock = threading.Lock()
        self._positions: Dict[str, float] = {}
        self._stamp: Dict[str, Any] = _zero_stamp()

        TopicSubscriber(broker, "/joint_states", self._on_joint_states)
        self._loop = PeriodicLoop("robot_state_publisher", PUBLISH_PERIOD_S, self._tick)

    def start(self) -> PeriodicLoop:
        return self._loop.start()

    # -- /joint_states ------------------------------------------------------

    def _on_joint_states(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        names = msg.get("name") or []
        values = msg.get("position") or []
        positions: Dict[str, float] = {}
        for name, value in zip(names, values):
            if isinstance(name, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    positions[name] = float(value)
        header = msg.get("header")
        stamp = header.get("stamp") if isinstance(header, dict) else None
        # Each message *replaces* the joint state: unnamed joints go back to 0.
        with self._lock:
            self._positions = positions
            self._stamp = dict(stamp) if isinstance(stamp, dict) else _zero_stamp()

    # -- loop ---------------------------------------------------------------

    def _tick(self) -> None:
        self._poll_description()
        self._publish_tf()

    def _poll_description(self) -> None:
        values, ok, _ = self._client.call("/param_server/get_param", {"name": "robot_description"})
        if not ok or not isinstance(values, dict):
            return  # never set yet
        version = values.get("version")
        if not isinstance(version, int) or version == self._processed_version:
            return
        self._processed_version = version

        error = ""
        try:
            robot = urdf.parse_robot_description(values.get("value"))
        except urdf.DescriptionError as exc:
            robot, error = None, str(exc) or "invalid description"
        except Exception as exc:  # noqa: BLE001 - a parser bug must reject, not crash
            log.exception("unexpected error parsing robot_description v%d", version)
            robot, error = None, f"internal error: {exc!r}"

        if robot is not None:
            self._robot = robot
            self._loaded_version = version
            log.info("loaded robot %r v%d: %d links, %d joints, root %r",
                     robot.name, version, len(robot.links), len(robot.joints), robot.root)
            self._publish_tf()  # the new robot is on /tf before the status says so
        else:
            log.info("rejected robot_description v%d: %s", version, error)

        status = {
            "version": version,
            "accepted": robot is not None,
            "error": error,
            "loaded_version": self._loaded_version,
            "root_link": self._robot.root if self._robot is not None else "",
        }
        self._client.call("/param_server/set_param", {"name": STATUS_PARAM, "value": status})

    def _publish_tf(self) -> None:
        robot = self._robot
        if robot is None:
            return
        with self._lock:
            positions = self._positions
            stamp = self._stamp
        transforms_out: List[Dict[str, Any]] = []
        for joint in robot.joints:
            q = positions.get(joint.name, 0.0) if joint.movable else 0.0
            translation, rotation = joint_transform(joint, q)
            transforms_out.append({
                "header": {"stamp": dict(stamp), "frame_id": joint.parent},
                "child_frame_id": joint.child,
                "transform": {
                    "translation": transforms.vec3_to_msg(translation),
                    "rotation": transforms.quat_to_msg(rotation),
                },
            })
        self._broker.publish("/tf", {"transforms": transforms_out})


def register(broker: Broker) -> List[PeriodicLoop]:
    return [RobotStatePublisher(broker).start()]
