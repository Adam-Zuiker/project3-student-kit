"""robot_world_state_publisher.py — `/tf` + `/global_pose` in,
`/xform_world` out (spec/FK_API.md, "robot_world_state_publisher").

Keeps only the most recent `/tf` message (never merges edges across
messages, or a previous robot's links would linger after a reload) and
the most recent global pose, then recomputes and republishes at 10 Hz.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from broker import Broker
from fk import transforms
from fk.kinematics import Edge, link_poses
from node_client import PeriodicLoop, TopicSubscriber

log = logging.getLogger("robot_world_state_publisher")

PUBLISH_PERIOD_S = 0.1
GLOBAL_FRAME = "global_frame"


class RobotWorldStatePublisher:
    def __init__(self, broker: Broker) -> None:
        self._broker = broker
        self._lock = threading.Lock()
        self._latest_tf: Optional[Dict[str, Any]] = None
        self._global_pose: Tuple[transforms.Vec3, transforms.Quat] = (
            (0.0, 0.0, 0.0),
            transforms.QUAT_IDENTITY,
        )
        TopicSubscriber(broker, "/tf", self._on_tf)
        TopicSubscriber(broker, "/global_pose", self._on_global_pose)
        self._loop = PeriodicLoop("robot_world_state_publisher", PUBLISH_PERIOD_S, self._tick)

    def start(self) -> PeriodicLoop:
        return self._loop.start()

    def _on_tf(self, msg: Any) -> None:
        if isinstance(msg, dict) and isinstance(msg.get("transforms"), list):
            with self._lock:
                self._latest_tf = msg

    def _on_global_pose(self, msg: Any) -> None:
        try:
            pose = (
                transforms.vec3_from_msg(msg["position"]),
                transforms.quat_normalize(transforms.quat_from_msg(msg["orientation"])),
            )
        except (KeyError, TypeError, ValueError):
            log.info("ignoring malformed /global_pose: %r", msg)
            return
        with self._lock:
            self._global_pose = pose

    def _tick(self) -> None:
        with self._lock:
            tf_msg = self._latest_tf
            position, orientation = self._global_pose
        if tf_msg is None or not tf_msg["transforms"]:
            return  # nothing received yet, or a joint-less robot (UNSPECIFIED)

        edges: List[Edge] = []
        for entry in tf_msg["transforms"]:
            t = entry["transform"]
            edges.append((
                entry["header"]["frame_id"],
                entry["child_frame_id"],
                transforms.mat4_from_translation_quat(
                    transforms.vec3_from_msg(t["translation"]),
                    transforms.quat_from_msg(t["rotation"]),
                ),
            ))
        stamp = tf_msg["transforms"][0]["header"]["stamp"]

        T_global_root = transforms.mat4_from_translation_quat(position, orientation)
        poses = link_poses(edges, T_global_root)
        self._broker.publish("/xform_world", {
            "transforms": [
                {
                    "header": {"stamp": dict(stamp), "frame_id": GLOBAL_FRAME},
                    "child_frame_id": link,
                    "matrix": transforms.mat4_to_column_major(M),
                }
                for link, M in poses.items()
            ]
        })


def register(broker: Broker) -> List[PeriodicLoop]:
    return [RobotWorldStatePublisher(broker).start()]
