#!/usr/bin/env python3
"""final_check.py — Black-box checks for the final project's categories C
(joint-state FK) and D (topic behavior, global pose), run against a live
`make run` over TCP, the way the autograder does.

    make run                          # terminal 1
    python3 tools/final_check.py      # terminal 2

Expected poses come from an independent oracle in this file (3x3 rotation
matrices + Rodrigues' formula), deliberately sharing no code with src/fk,
so a bug there cannot make both sides agree.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
import itertools

TOL = 1e-4
PERIOD_BOUND_S = 0.2  # one 5 Hz period

# ---------------------------------------------------------------------------
# Oracle math (independent of src/fk)
# ---------------------------------------------------------------------------


def matmul(A, B):
    return [[sum(A[r][k] * B[k][c] for k in range(len(B))) for c in range(len(B[0]))] for r in range(len(A))]


def rot_axis(axis, angle):
    """Rodrigues: R = I + sinθ K + (1 − cosθ) K²."""
    n = math.sqrt(sum(a * a for a in axis))
    x, y, z = (a / n for a in axis)
    K = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    K2 = matmul(K, K)
    s, c = math.sin(angle), 1 - math.cos(angle)
    return [[(1 if r == col else 0) + s * K[r][col] + c * K2[r][col] for col in range(3)] for r in range(3)]


def rot_rpy(r, p, y):
    return matmul(rot_axis((0, 0, 1), y), matmul(rot_axis((0, 1, 0), p), rot_axis((1, 0, 0), r)))


def hom(R, t):
    return [R[0] + [t[0]], R[1] + [t[1]], R[2] + [t[2]], [0, 0, 0, 1]]


def rot_quat(x, y, z, w):
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def oracle(joints, positions, T_global_root):
    """joints: list of (name, type, parent, child, xyz, rpy, axis). Returns {link: 4x4}."""
    by_parent = {}
    children = set()
    for j in joints:
        by_parent.setdefault(j[2], []).append(j)
        children.add(j[3])
    root = next(j[2] for j in joints if j[2] not in children)
    poses = {root: T_global_root}
    stack = [root]
    while stack:
        parent = stack.pop()
        for name, jtype, _, child, xyz, rpy, axis in by_parent.get(parent, []):
            q = positions.get(name, 0.0) if jtype != "fixed" else 0.0
            T = hom(rot_rpy(*rpy), xyz)
            if jtype in ("revolute", "continuous"):
                T = matmul(T, hom(rot_axis(axis, q), (0, 0, 0)))
            elif jtype == "prismatic":
                T = matmul(T, hom(I3, [q * a for a in axis]))
            poses[child] = matmul(poses[parent], T)
            stack.append(child)
    return poses


def from_column_major(m):
    return [[m[4 * c + r] for c in range(4)] for r in range(4)]


def rotation_angle_between(Ra, Rb):
    """Geodesic angle of Ra^T Rb (the spec's rotation metric)."""
    tr = sum(Ra[k][i] * Rb[k][i] for i in range(3) for k in range(3))
    return math.acos(max(-1.0, min(1.0, (tr - 1) / 2)))


# ---------------------------------------------------------------------------
# Test robots
# ---------------------------------------------------------------------------

PI = math.pi
# (name, type, parent, child, xyz, rpy, axis) — mirrors ARM_URDF below exactly.
ARM_JOINTS = [
    ("j1", "revolute", "base", "l1", (0, 0, 1), (0, 0, 0), (1, 1, 0)),
    ("j2", "continuous", "l1", "l2", (1, 0, 0), (PI / 2, 0, 0), (0, 0, 2)),
    ("j3", "prismatic", "l2", "l3", (0, 0.5, 0), (0, 0, PI / 2), (2, 0, 0)),
    ("cam", "fixed", "base", "camera", (0.1, 0, 0), (0, PI / 2, 0), (1, 0, 0)),
]
ARM_URDF = f"""<?xml version="1.0"?>
<robot name="final_check_arm">
  <link name="l3"/><link name="l2"/><link name="l1"/><link name="camera"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 1"/><axis xyz="1 1 0"/><limit lower="-3" upper="3"/></joint>
  <joint name="j2" type="continuous"><parent link="l1"/><child link="l2"/>
    <origin xyz="1 0 0" rpy="{PI / 2!r} 0 0"/><axis xyz="0 0 2"/></joint>
  <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/>
    <origin xyz="0 0.5 0" rpy="0 0 {PI / 2!r}"/><axis xyz="2 0 0"/><limit lower="0" upper="1"/></joint>
  <joint name="cam" type="fixed"><parent link="base"/><child link="camera"/>
    <origin xyz="0.1 0 0" rpy="0 {PI / 2!r} 0"/></joint>
  <link name="base"/>
</robot>
"""
OTHER_URDF = """<robot name="other"><link name="chassis"/><link name="wheel"/>
  <joint name="axle" type="continuous"><parent link="chassis"/><child link="wheel"/><axis xyz="0 1 0"/></joint>
</robot>"""

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class Client:
    def __init__(self, port=9095):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.stream = self.sock.makefile("rwb")
        self.ids = itertools.count()

    def send(self, message):
        self.stream.write((json.dumps(message) + "\n").encode())
        self.stream.flush()

    def _lines(self):
        for line in self.stream:
            yield json.loads(line)
        raise ConnectionError("runtime closed the connection")

    def call(self, service, args):
        call_id = f"fc-{next(self.ids)}"
        self.send({"op": "call_service", "service": service, "id": call_id, "args": args})
        for m in self._lines():
            if m.get("op") == "service_response" and m.get("id") == call_id:
                return m

    def subscribe(self, *topics):
        for t in topics:
            self.send({"op": "subscribe", "topic": t})

    def publish(self, topic, msg):
        self.send({"op": "advertise", "topic": topic})
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def next_msg(self, topic, timeout=3.0, where=lambda m: True):
        deadline = time.monotonic() + timeout
        for m in self._lines():
            if m.get("op") == "publish" and m.get("topic") == topic and where(m["msg"]):
                return m["msg"]
            if time.monotonic() > deadline:
                return None

    def load(self, urdf):
        version = self.call("/param_server/set_param", {"name": "robot_description", "value": urdf})["values"]["version"]
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            st = self.call("/param_server/get_param", {"name": "/robot_state_publisher/description_status"})["values"]["value"]
            if st and st["version"] >= version:
                return st
            time.sleep(0.05)
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

results = []


def check(label, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail and not ok else ""))


def joint_states(names, positions, sec, nanosec):
    return {"header": {"stamp": {"sec": sec, "nanosec": nanosec}, "frame_id": ""},
            "name": names, "position": positions}


def compare_world(msg, expected, stamp=None):
    """Returns '' if msg matches expected within tolerance, else a reason."""
    got = {e["child_frame_id"]: e for e in msg["transforms"]}
    if set(got) - {"global_frame"} != set(expected):
        return f"links {sorted(got)} != {sorted(expected)}"
    for link, E in expected.items():
        e = got[link]
        if e["header"]["frame_id"] != "global_frame":
            return f"{link}: frame_id {e['header']['frame_id']!r}"
        if stamp is not None and e["header"]["stamp"] != stamp:
            return f"{link}: stamp {e['header']['stamp']} != {stamp}"
        M = from_column_major(e["matrix"])
        dt = max(abs(M[r][3] - E[r][3]) for r in range(3))
        if dt > TOL:
            return f"{link}: translation off by {dt:.2e}"
        R = [row[:3] for row in M[:3]]
        da = rotation_angle_between(R, [row[:3] for row in E[:3]])
        if da > TOL:
            return f"{link}: rotation off by {da:.2e} rad"
        RtR = matmul([list(c) for c in zip(*R)], R)
        if any(abs(RtR[i][j] - (i == j)) > TOL for i in range(3) for j in range(3)):
            return f"{link}: rotation block not orthonormal"
        if any(abs(M[3][c] - (c == 3)) > 1e-9 for c in range(4)):
            return f"{link}: bottom row {M[3]}"
    return ""


def wait_world(c, expected, stamp=None, timeout=3.0):
    """Wait until /xform_world matches; returns '' or the last mismatch."""
    reason = ["no /xform_world received"]

    def matches(msg):
        reason[0] = compare_world(msg, expected, stamp)
        return reason[0] == ""

    c.next_msg("/xform_world", timeout, matches)
    return reason[0]


def main():
    port = int(os.environ.get("FK_PORT", "9095"))
    c = Client(port)
    identity = hom(I3, (0, 0, 0))
    print("Setup")
    c.publish("/global_pose", {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}})
    st = c.load(ARM_URDF)
    check("test arm accepted, root is 'base' (not the first link)", bool(st and st["accepted"] and st["root_link"] == "base"), str(st))
    c.subscribe("/tf", "/xform_world")

    print("C. Joint-state FK")
    pos = {"j1": 0.7, "j2": 5.5, "j3": 0.4}
    c.publish("/joint_states", joint_states(["j1", "j2", "j3", "cam", "ghost"], [0.7, 5.5, 0.4, 9.0, 3.0], 123, 456))
    check("revolute (1,1,0) axis, continuous past π, prismatic 2x axis under yaw; fixed + unknown names ignored",
          not (r := wait_world(c, oracle(ARM_JOINTS, pos, identity), {"sec": 123, "nanosec": 456})), r)
    tf = c.next_msg("/tf")
    check("/tf has one entry per joint incl. fixed, all with the verbatim stamp",
          tf is not None and len(tf["transforms"]) == 4
          and all(e["header"]["stamp"] == {"sec": 123, "nanosec": 456} for e in tf["transforms"])
          and all(abs(math.sqrt(sum(e["transform"]["rotation"][k] ** 2 for k in "xyzw")) - 1) < TOL for e in tf["transforms"]),
          json.dumps(tf)[:300] if tf else "no /tf")

    c.publish("/joint_states", joint_states(["j2"], [-6.0], 124, 0))
    check("a message naming only j2 sends j1 and j3 back to 0",
          not (r := wait_world(c, oracle(ARM_JOINTS, {"j2": -6.0}, identity), {"sec": 124, "nanosec": 0})), r)

    c.publish("/joint_states", joint_states(["j2"], [-6.0], 124, 0))  # duplicate delivery is legal
    check("duplicate message is harmless",
          not (r := wait_world(c, oracle(ARM_JOINTS, {"j2": -6.0}, identity))), r)

    print("D. Topic behavior and global pose")
    axis, ang = (0.3, 0.2, 0.9), 1.1
    n = math.sqrt(sum(a * a for a in axis))
    qv = [a / n * math.sin(ang / 2) for a in axis]
    c.publish("/global_pose", {"position": {"x": 1, "y": -2, "z": 0.5},
                               "orientation": {"x": qv[0], "y": qv[1], "z": qv[2], "w": math.cos(ang / 2)}, "seq": 7})
    G = hom(rot_axis(axis, ang), (1, -2, 0.5))
    check("/global_pose (translation + general rotation, extra field) moves every link, root included",
          not (r := wait_world(c, oracle(ARM_JOINTS, {"j2": -6.0}, G))), r)

    counts, complete = {"/tf": 0, "/xform_world": 0}, True
    t0 = time.monotonic()
    for m in c._lines():
        if m.get("op") == "publish" and m["topic"] in counts:
            counts[m["topic"]] += 1
            expected = 4 if m["topic"] == "/tf" else 5
            complete &= len([e for e in m["msg"]["transforms"] if e["child_frame_id"] != "global_frame"]) == expected
        if time.monotonic() - t0 > 3.0:
            break
    check(f"rate with nothing changing over 3 s: {counts} (need >= 15 each), every message complete",
          min(counts.values()) >= 15 and complete)

    late = Client(port)
    t0 = time.monotonic()
    late.subscribe("/tf", "/xform_world")
    first_tf = late.next_msg("/tf", timeout=2.0)
    first_world = late.next_msg("/xform_world", timeout=2.0)
    waited = time.monotonic() - t0
    check(f"late subscriber gets complete /tf and /xform_world ({waited:.2f} s, bound ~{2 * PERIOD_BOUND_S:.1f} s)",
          first_tf is not None and len(first_tf["transforms"]) == 4 and first_world is not None
          and not compare_world(first_world, oracle(ARM_JOINTS, {"j2": -6.0}, G)) and waited < 2 * PERIOD_BOUND_S + 0.2)

    print("Reload")
    st = c.load(OTHER_URDF)
    check("different robot accepted", bool(st and st["accepted"] and st["root_link"] == "chassis"), str(st))
    other = [("axle", "continuous", "chassis", "wheel", (0, 0, 0), (0, 0, 0), (0, 1, 0))]
    # Joint state across a reload is UNSPECIFIED, so publish a fresh one before comparing poses.
    c.publish("/joint_states", joint_states(["axle"], [1.0], 200, 0))
    check("reload replaces: /xform_world has only chassis + wheel, at the right poses",
          not (r := wait_world(c, oracle(other, {"axle": 1.0}, G))), r)
    tf = c.next_msg("/tf", where=lambda m: m["transforms"] and m["transforms"][0]["header"]["stamp"]["sec"] == 200)
    check("reload replaces: /tf has only the axle", tf is not None and [e["child_frame_id"] for e in tf["transforms"]] == ["wheel"])

    st = c.load("<robot><link name='broken'/><link name='two_roots'/></robot>")
    check("invalid reload rejected, previous robot stays", bool(st and not st["accepted"] and st["root_link"] == "chassis"), str(st))

    print("Line size")
    big = ARM_URDF.replace("<link name=\"base\"/>", "<link name=\"base\"/><!--" + "x" * 1_000_000 + "-->")
    st = c.load(big)
    check(f"~{len(json.dumps(big)) // 1024} KiB description accepted", bool(st and st["accepted"]), str(st))

    c.publish("/global_pose", {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}})
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
