"""Shared test helpers. Tests import `src/` directly, no runtime needed."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fk.transforms import quat_multiply  # noqa: E402

ROBOTS_DIR = os.path.join(os.path.dirname(__file__), "robots")
TOL = 1e-9


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def sandwich(q, v):
    """Rotate v by unit quaternion q as q ⊗ (0, v) ⊗ q*: built only on quat_multiply."""
    r = quat_multiply(quat_multiply(q, (0.0, *v)), conj(q))
    return r[1:]


def same_rotation(testcase, q1, q2, tol=1e-9):
    """q and -q are the same rotation."""
    dot = abs(sum(a * b for a, b in zip(q1, q2)))
    testcase.assertAlmostEqual(dot, 1.0, delta=tol, msg=f"{q1} vs {q2}")


def assert_vec(testcase, a, b, tol=1e-9):
    for x, y in zip(a, b):
        testcase.assertAlmostEqual(x, y, delta=tol, msg=f"{tuple(a)} vs {tuple(b)}")


def read_robot(name):
    with open(os.path.join(ROBOTS_DIR, name), encoding="utf-8") as f:
        return f.read()


H = math.sqrt(0.5)  # cos 45° = sin 45°
