"""joint_state_publisher.py — Holds the robot's joint positions, servos them
toward `/joint_trajectory` setpoints, publishes `/joint_states` at 10 Hz
(spec/FK_API.md, "Recommended nodes"). Started by `make demo` only, never
by `make run`: during grading `/joint_states` must come from outside.

Same one-loop-thread shape as robot_state_publisher: the loop polls
`robot_description`, steps the servo, and publishes. The `/joint_trajectory`
callback and the reset service only write setpoints under the lock.

Setpoints are kept by joint *name*, including names the loaded robot does
not have (yet): a trajectory can arrive just before this node has picked up
the robot it was meant for, and is applied once that robot loads.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import servo
from broker import Broker
from fk import urdf
from node_client import PeriodicLoop, ServiceClient, TopicSubscriber
from services.provider import register as register_provider

log = logging.getLogger("joint_state_publisher")

PUBLISH_PERIOD_S = 0.1


def _rest_position(joint: urdf.Joint) -> float:
    """Middle of the limits, or 0 for continuous joints (the reset pose)."""
    if joint.type in ("revolute", "prismatic"):
        return (joint.lower + joint.upper) / 2.0
    return 0.0


class JointStatePublisher:
    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._client = ServiceClient(broker, "joint_state_publisher")

        # Loop-thread-only state.
        self._robot: Optional[urdf.RobotModel] = None
        self._processed_version = 0
        self._positions: Dict[str, float] = {}
        self._last_tick: Optional[float] = None

        # Shared with the /joint_trajectory callback and the reset service.
        self._lock = threading.Lock()
        self._setpoints: Dict[str, float] = {}

        TopicSubscriber(broker, "/joint_trajectory", self._on_joint_trajectory)
        self._reset_provider = register_provider(broker, "/joint_state_publisher/reset", self._reset)
        self._loop = PeriodicLoop("joint_state_publisher", PUBLISH_PERIOD_S, self._tick)

    def start(self) -> List[Any]:
        return [self._reset_provider, self._loop.start()]

    # -- inputs -----------------------------------------------------------

    def _on_joint_trajectory(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        names = msg.get("joint_names") or []
        points = msg.get("points") or []
        if not points or not isinstance(points[-1], dict):
            return
        positions = points[-1].get("positions") or []
        update = {
            n: float(p) for n, p in zip(names, positions)
            if isinstance(n, str) and isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p)
        }
        with self._lock:
            self._setpoints.update(update)

    def _reset(self, args: Any) -> Tuple[Any, bool, str]:
        robot = self._robot
        if robot is None:
            return {}, False, "no robot loaded"
        rest = {j.name: _rest_position(j) for j in robot.joints if j.movable}
        with self._lock:
            self._setpoints.update(rest)
        return {"setpoints": rest}, True, ""

    # -- loop ---------------------------------------------------------------

    def _tick(self) -> None:
        self._poll_description()
        robot = self._robot
        if robot is None:
            return
        now = time.monotonic()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        with self._lock:
            setpoints = dict(self._setpoints)
        names, positions = [], []
        for joint in robot.joints:
            if not joint.movable:
                continue
            q = self._positions.get(joint.name, 0.0)
            target = setpoints.get(joint.name, q)
            q = servo.servo_step(joint, q, target, dt)
            self._positions[joint.name] = q
            names.append(joint.name)
            positions.append(q)

        wall = time.time()
        self._broker.publish("/joint_states", {
            "header": {"stamp": {"sec": int(wall), "nanosec": int((wall % 1) * 1e9)}, "frame_id": ""},
            "name": names,
            "position": positions,
            "velocity": [],
            "effort": [],
        })

    def _poll_description(self) -> None:
        values, ok, _ = self._client.call("/param_server/get_param", {"name": "robot_description"})
        if not ok or not isinstance(values, dict):
            return
        version = values.get("version")
        if not isinstance(version, int) or version == self._processed_version:
            return
        self._processed_version = version
        try:
            robot = urdf.parse_robot_description(values.get("value"))
        except Exception:  # noqa: BLE001 - robot_state_publisher reports why; just keep the old robot
            return
        # A joint the old and new robot share keeps its position, clamped to
        # the new limits; a new joint starts at 0 (clamped).
        self._positions = {
            j.name: servo.clamp_to_limits(j, self._positions.get(j.name, 0.0))
            for j in robot.joints if j.movable
        }
        self._robot = robot
        log.info("servoing %d movable joints of %r", len(self._positions), robot.name)


def register(broker: Broker) -> List[Any]:
    return JointStatePublisher(broker).start()
