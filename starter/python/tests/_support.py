"""Shared helpers for the FK and FSM tests.

The reference math here is written independently of src/ so that a bug in
src/ cannot also corrupt the expected values.
"""

from __future__ import annotations

import copy
import math
import os
import random
import sys
import threading
import types

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(TESTS_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

TOL = 1e-9
H = math.sqrt(0.5)
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def flat(x):
    if isinstance(x, (list, tuple)):
        return [v for item in x for v in flat(item)]
    return [float(x)]


def close(actual, expected, what, tol=TOL):
    a, e = flat(actual), flat(expected)
    assert len(a) == len(e), f"{what}: shape differs, got {actual!r}, expected {expected!r}"
    for x, y in zip(a, e):
        assert abs(x - y) <= tol, f"{what}: got {actual!r}, expected {expected!r}"


def same_rotation(q1, q2, what, tol=1e-9):
    assert len(q1) == 4, f"{what}: not a 4-element quaternion: {q1!r}"
    n1 = math.sqrt(sum(c * c for c in q1))
    n2 = math.sqrt(sum(c * c for c in q2))
    d = abs(sum(a * b for a, b in zip(q1, q2))) / (n1 * n2)
    assert abs(n1 - 1.0) <= 1e-9, f"{what}: not unit length (|q|={n1}): {q1!r}"
    assert abs(d - 1.0) <= tol, f"{what}: got {q1!r}, expected {q2!r} (or its negative)"


def run_with_timeout(fn, seconds, what):
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"{what}: did not finish within {seconds} s (infinite loop?)"
    if "error" in box:
        raise box["error"]
    return box["value"]


# ---------------------------------------------------------------------------
# Reference math (independent of src/)
# ---------------------------------------------------------------------------


def ref_qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def ref_rotate(q, v):
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return ref_qmul(ref_qmul(q, (0.0, *v)), conj(q))[1:]


def quat_matrix_ref(q):
    cols = [ref_rotate(q, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return [[cols[c][r] for c in range(3)] for r in range(3)]


def ref_matmul(A, B):
    return [[sum(A[r][k] * B[k][c] for k in range(len(B))) for c in range(len(B[0]))]
            for r in range(len(A))]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def ref_rpy_matrix(roll, pitch, yaw):
    return ref_matmul(rot_z(yaw), ref_matmul(rot_y(pitch), rot_x(roll)))


def axis_angle_matrix(axis, angle):
    """Rodrigues: R = cI + s[k]x + (1-c)kk^T, with k the unit axis."""
    n = math.sqrt(sum(a * a for a in axis))
    kx, ky, kz = (a / n for a in axis)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [
        [c + t * kx * kx, t * kx * ky - s * kz, t * kx * kz + s * ky],
        [t * ky * kx + s * kz, c + t * ky * ky, t * ky * kz - s * kx],
        [t * kz * kx - s * ky, t * kz * ky + s * kx, c + t * kz * kz],
    ]


def mat_vec(R, v):
    return [sum(R[r][c] * v[c] for c in range(3)) for r in range(3)]


def homogeneous(R, t):
    return [list(R[0]) + [t[0]], list(R[1]) + [t[1]], list(R[2]) + [t[2]], [0.0, 0.0, 0.0, 1.0]]


def translation_of(M):
    return [M[0][3], M[1][3], M[2][3]]


def make_rng(seed):
    return random.Random(seed)


def random_quat(rng, unit=True):
    q = tuple(rng.uniform(-1, 1) for _ in range(4))
    if not unit:
        return q
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def random_vec(rng, lo=-2.0, hi=2.0):
    return tuple(rng.uniform(lo, hi) for _ in range(3))


def random_rigid(rng):
    return homogeneous(quat_matrix_ref(random_quat(rng)), random_vec(rng))


# ---------------------------------------------------------------------------
# Robot fixtures
# ---------------------------------------------------------------------------

TWO_LINK_ARM = """<?xml version="1.0"?>
<robot name="two_link_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <link name="base_link"/>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <link name="upper_arm">
    <visual><geometry><cylinder radius="0.05" length="0.5"/></geometry></visual>
  </link>
  <joint name="elbow_mount" type="fixed">
    <parent link="upper_arm"/>
    <child link="forearm"/>
    <origin xyz="0.5 0 0" rpy="0 1.5707963267948966 0"/>
  </joint>
  <link name="forearm"/>
  <gazebo reference="forearm"><material>Gazebo/Grey</material></gazebo>
</robot>
"""

BRANCHING_TREE = """<?xml version="1.0" encoding="UTF-8"?>
<robot name="branching_tree">
  <joint name="left_shoulder" type="continuous">
    <parent link="torso"/>
    <child link="left_arm"/>
    <origin xyz="0 0.2 0.5"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="right_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="right_arm"/>
    <origin xyz="0 -0.2 0.5" rpy="0 0 3.141592653589793"/>
    <axis xyz="0 2 0"/>
    <limit lower="-1" upper="1"/>
  </joint>
  <joint name="left_hand_slide" type="prismatic">
    <parent link="left_arm"/>
    <child link="left_hand"/>
    <origin xyz="0.3 0 0" rpy="0 0 1.5707963267948966"/>
    <axis xyz="2 0 0"/>
    <limit lower="0" upper="0.5"/>
  </joint>
  <link name="left_hand"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="left_arm"/>
  <link name="right_arm"><visual><geometry><sphere radius=".05"/></geometry></visual></link>
  <link name="torso"/>
  <transmission name="t"><joint name="hollow_joint_not_a_robot_joint"/></transmission>
</robot>
"""


# ---------------------------------------------------------------------------
# FSM fixtures
# ---------------------------------------------------------------------------

_SPEC_FSM = {
    "initial_state": "home",
    "states": [
        {"name": "home",
         "target": {"joint_names": ["shoulder", "elbow"], "positions": [0.0, 0.0]},
         "epsilon": 0.02,
         "on_arrival": "to_wave",
         "transitions": [{"name": "to_wave", "next_state": "wave"}]},
        {"name": "wave",
         "target": {"joint_names": ["shoulder", "elbow"], "positions": [0.8, -0.5]},
         "epsilon": 0.02,
         "on_arrival": "to_home",
         "transitions": [{"name": "to_home", "next_state": "home"}]},
    ],
}


def spec_fsm():
    return copy.deepcopy(_SPEC_FSM)


def make_fsm():
    """A real FiniteStateMachine on a real Broker, never started, with every
    publish captured in `published` as (topic, msg)."""
    from broker import Broker
    from finite_state_machine import FiniteStateMachine

    broker = Broker()
    published = []
    broker.publish = lambda topic, msg: published.append((topic, copy.deepcopy(msg)))
    return FiniteStateMachine(broker), published


def messages_on(published, topic):
    return [msg for t, msg in published if t == topic]


def joint_states(names, positions):
    return {"name": list(names), "position": list(positions)}


class FakeClock:
    """Replaces finite_state_machine's view of time.monotonic()."""

    def __init__(self, start=1000.0):
        self.now = start

    def advance(self, seconds):
        self.now += seconds

    def install(self, module):
        module.time = types.SimpleNamespace(monotonic=lambda: self.now)
        return self
