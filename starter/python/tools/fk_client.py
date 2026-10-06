#!/usr/bin/env python3
"""fk_client.py — Poke a running runtime (`make run`) the way the autograder
does: as an external TCP/JSON client on 127.0.0.1:9095.

    python3 tools/fk_client.py tests/robots/two_link_arm.urdf
    python3 tools/fk_client.py tests/robots/two_link_arm.urdf --joint shoulder=1.57
    python3 tools/fk_client.py ../../robots/fetch/fetch.urdf --global-pose 1 2 0 90
    python3 tools/fk_client.py --tf          # just watch the current /tf
    python3 tools/fk_client.py --trajectory shoulder_pan_joint=1.0   # make demo only

Prints the description status, then one /xform_world message (link, xyz).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import socket
import sys
import time


class Client:
    def __init__(self, host="127.0.0.1", port=9095):
        self.sock = socket.create_connection((host, port))
        self.stream = self.sock.makefile("rwb")
        self.ids = itertools.count()

    def send(self, message):
        self.stream.write((json.dumps(message) + "\n").encode())
        self.stream.flush()

    def call(self, service, args):
        call_id = f"call-{next(self.ids)}"
        self.send({"op": "call_service", "service": service, "id": call_id, "args": args})
        for line in self.stream:
            reply = json.loads(line)
            if reply.get("op") == "service_response" and reply.get("id") == call_id:
                return reply
        raise ConnectionError("runtime closed the connection")

    def publish(self, topic, type_, msg):
        self.send({"op": "advertise", "topic": topic, "type": type_})
        self.send({"op": "publish", "topic": topic, "msg": msg})

    def next_message(self, topic, type_):
        self.send({"op": "subscribe", "topic": topic, "type": type_})
        for line in self.stream:
            message = json.loads(line)
            if message.get("op") == "publish" and message.get("topic") == topic:
                return message["msg"]
        raise ConnectionError("runtime closed the connection")


def load(client, path):
    with open(path, encoding="utf-8") as f:
        reply = client.call("/param_server/set_param", {"name": "robot_description", "value": f.read()})
    version = reply["values"]["version"]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        status = client.call("/param_server/get_param",
                             {"name": "/robot_state_publisher/description_status"})["values"]["value"]
        if status and status["version"] >= version:
            return status
        time.sleep(0.05)
    sys.exit(f"no description status for version {version} within 2 s (spec bound)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("urdf", nargs="?", help="URDF file to load (omit to use the loaded robot)")
    p.add_argument("--joint", action="append", default=[], metavar="NAME=POS",
                   help="publish /joint_states with these positions (repeatable)")
    p.add_argument("--trajectory", action="append", default=[], metavar="NAME=POS",
                   help="publish a /joint_trajectory setpoint (needs make demo's joint_state_publisher)")
    p.add_argument("--global-pose", nargs=4, type=float, metavar=("X", "Y", "Z", "YAW_DEG"))
    p.add_argument("--tf", action="store_true", help="print one /tf message instead of /xform_world")
    args = p.parse_args()

    client = Client()
    if args.urdf:
        print("status:", json.dumps(load(client, args.urdf)))

    if args.joint:
        names, positions = zip(*((n, float(v)) for n, v in (s.split("=", 1) for s in args.joint)))
        client.publish("/joint_states", "sensor_msgs/JointState", {
            "header": {"stamp": {"sec": 42, "nanosec": 7}, "frame_id": ""},
            "name": list(names), "position": list(positions)})
    if args.trajectory:
        names, positions = zip(*((n, float(v)) for n, v in (s.split("=", 1) for s in args.trajectory)))
        client.publish("/joint_trajectory", "trajectory_msgs/JointTrajectory", {
            "joint_names": list(names), "points": [{"positions": list(positions)}]})
    if args.global_pose:
        x, y, z, yaw = args.global_pose
        half = math.radians(yaw) / 2
        client.publish("/global_pose", "geometry_msgs/Pose", {
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}})
    if args.trajectory:
        time.sleep(1.5)  # let the servo get there
    elif args.joint or args.global_pose:
        time.sleep(0.3)  # let it reach /tf and then /xform_world

    if args.tf:
        for e in client.next_message("/tf", "tf/tfMessage")["transforms"]:
            t, r = e["transform"]["translation"], e["transform"]["rotation"]
            print(f'{e["header"]["frame_id"]:>24} -> {e["child_frame_id"]:<24} '
                  f'xyz=({t["x"]:+.4f} {t["y"]:+.4f} {t["z"]:+.4f}) '
                  f'q_xyzw=({r["x"]:+.4f} {r["y"]:+.4f} {r["z"]:+.4f} {r["w"]:+.4f})')
    else:
        msg = client.next_message("/xform_world", "xform_world/MatrixTransformArray")
        for e in sorted(msg["transforms"], key=lambda e: e["child_frame_id"]):
            m = e["matrix"]
            print(f'{e["child_frame_id"]:<32} xyz=({m[12]:+.4f} {m[13]:+.4f} {m[14]:+.4f})')
        print("stamp:", msg["transforms"][0]["header"]["stamp"] if msg["transforms"] else None)


if __name__ == "__main__":
    main()
