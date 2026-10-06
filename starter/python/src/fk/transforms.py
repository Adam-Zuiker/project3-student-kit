"""fk/transforms.py — 3D rigid-body math: unit quaternions and 4x4
homogeneous transforms.

Conventions used everywhere in this package:

  - `Vec3`  is a tuple `(x, y, z)`.
  - `Quat`  is a tuple `(w, x, y, z)`, **scalar first**. (ROS messages put
            `w` last; `quat_to_msg` / `quat_from_msg` do that reordering at
            the edge, so nothing inside `fk/` ever has to think about it.)
  - `Mat3` / `Mat4` are row-major nested lists: `M[row][col]`.
            `mat4_to_column_major` produces the flat `/xform_world` layout.

`q` and `-q` are the same rotation; nothing here depends on the sign.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)
Mat3 = List[List[float]]
Mat4 = List[List[float]]

QUAT_IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def quat_multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product `a ⊗ b`: the rotation that applies `b` first, then `a`
    (when both rotate vectors expressed in the same, fixed frame)."""
    s1, s2 = a[0], b[0]
    v1, v2 = list(a[1:]), list(b[1:])

    ijk_vec = add(add(scale(v2, s1), scale(v1, s2)), cross(v1, v2))
    return (s1 * s2 - dot(v1, v2), *ijk_vec)

def dot(a: list, b: list):
    if len(a) != len(b):
        raise ValueError("List lengths are not the same")
    dot_product = 0
    for i in range(len(a)):
        dot_product += a[i] * b[i]
    return dot_product

def cross(a: list, b: list):
    return [
        a[1] * b[2] - a[2] * b[1], 
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]

def add(a: list, b: list):
    return [i + j for i,j in zip(a,b)]

def scale(v: list, s: float):
    return [x * s for x in v]


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    """Rotation of `angle` radians about `axis`. `axis` need not be unit
    length; normalize it here. A zero-length axis has no defined rotation
    (UNSPECIFIED by the spec): return the identity rather than dividing by 0."""
    if not axis[0] and not axis[1] and not axis[2]:
            return [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
    axis = axis_normalize(axis)
    q_w = math.cos(angle / 2)
    q_x = axis[0] * math.sin(angle / 2)
    q_y = axis[1] * math.sin(angle / 2)
    q_z = axis[2] * math.sin(angle / 2)
    return (q_w, q_x, q_y, q_z)

def axis_normalize(axis: Vec3):
    norm = (axis[0]**2 + axis[1]**2 + axis[2]**2)**0.5
    return [i / norm for i in axis]

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    """URDF fixed-axis roll-pitch-yaw: `R = Rz(yaw) · Ry(pitch) · Rx(roll)`."""
    cr, cp, cy = math.cos(roll / 2), math.cos(pitch / 2), math.cos(yaw / 2)
    sr, sp, sy = math.sin(roll / 2), math.sin(pitch / 2), math.sin(yaw / 2)
    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy
    return (q_w, q_x, q_y, q_z)

def quat_to_rotation_matrix(q: Quat) -> Mat3:
    """The 3x3 rotation matrix of a unit quaternion."""
    q = quat_normalize(q)
    w, x, y, z = q
    return [
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ]


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(sum(c * c for c in q))
    if n == 0.0:
        return QUAT_IDENTITY
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def rotate_vector(q: Quat, v: Vec3) -> Vec3:
    R = quat_to_rotation_matrix(q)
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


# ---------------------------------------------------------------------------
# 4x4 homogeneous transforms
# ---------------------------------------------------------------------------


def mat4_identity() -> Mat4:
    return [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]


def mat4_from_translation_quat(t: Vec3, q: Quat) -> Mat4:
    """`Trans(t) · Rot(q)` as one 4x4 matrix."""
    R = quat_to_rotation_matrix(q)
    return [
        [R[0][0], R[0][1], R[0][2], t[0]],
        [R[1][0], R[1][1], R[1][2], t[1]],
        [R[2][0], R[2][1], R[2][2], t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_multiply(A: Mat4, B: Mat4) -> Mat4:
    """`A · B`: express B's frame in A's parent frame."""
    B_T = tranpose(B)
    B_T_in_A = []
    for c in range(len(B)):
        transformed = []
        for r in range(len(A)):
            transformed.append(dot(B_T[c], A[r]))
        B_T_in_A.append(transformed)
    return tranpose(B_T_in_A)


def mat4_to_column_major(M: Mat4) -> List[float]:
    """Flatten for `/xform_world`: `out[4*c + r] = M[r][c]`."""
    return [float(M[r][c]) for c in range(4) for r in range(4)]

def tranpose(A: Mat4):
    T = []
    for c in range(len(A)):
        T.append([A[r][c] for r in range(len(A))])
    return T


# ---------------------------------------------------------------------------
# Wire-format adapters (ROS dicts <-> tuples)
# ---------------------------------------------------------------------------


def vec3_to_msg(v: Sequence[float]) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def quat_to_msg(q: Quat) -> dict:
    return {"x": float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": float(q[0])}


def vec3_from_msg(m: dict) -> Vec3:
    return (float(m["x"]), float(m["y"]), float(m["z"]))


def quat_from_msg(m: dict) -> Quat:
    return (float(m["w"]), float(m["x"]), float(m["y"]), float(m["z"]))
