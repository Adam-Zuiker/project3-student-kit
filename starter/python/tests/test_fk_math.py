"""Quaternions and 4x4 transforms (src/fk/transforms.py), checked against
independent reference math in _support.py."""

from __future__ import annotations

import math

from _support import (
    H, IDENTITY_Q, axis_angle_matrix, close, homogeneous, make_rng, mat_vec,
    quat_matrix_ref, random_quat, random_vec, ref_matmul, ref_qmul, ref_rotate,
    ref_rpy_matrix, same_rotation, translation_of,
)
from fk.transforms import (
    add, cross, dot, mat4_from_translation_quat, mat4_identity, mat4_multiply,
    mat4_to_column_major, quat_from_axis_angle, quat_from_msg, quat_from_rpy,
    quat_multiply, quat_normalize, quat_to_msg, quat_to_rotation_matrix,
    rotate_vector, scale, tranpose, vec3_from_msg, vec3_to_msg,
)


def test_vector_helpers_dot_cross_add_scale_transpose():
    assert dot([1, 2, 3], [4, -5, 6]) == 12, f"dot([1,2,3],[4,-5,6]) = {dot([1, 2, 3], [4, -5, 6])}, expected 12"
    close(cross([1, 0, 0], [0, 1, 0]), [0, 0, 1], "cross(x, y) must be +z (right-handed)")
    close(cross([0, 1, 0], [0, 0, 1]), [1, 0, 0], "cross(y, z) must be +x")
    close(cross([2, 3, 4], [5, 6, 7]), [-3, 6, -3], "cross([2,3,4],[5,6,7])")
    close(add([1, 2, 3], [10, 20, 30]), [11, 22, 33], "add")
    close(scale([1, -2, 3], 2.5), [2.5, -5, 7.5], "scale")
    M = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    close(tranpose(M), [[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]], "tranpose")


def test_quat_multiply_hamilton_basis_rules():
    one, i, j, k = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
    close(quat_multiply(i, j), k, "i*j must be k")
    close(quat_multiply(j, k), i, "j*k must be i")
    close(quat_multiply(k, i), j, "k*i must be j")
    close(quat_multiply(j, i), (0, 0, 0, -1), "j*i must be -k")
    close(quat_multiply(i, i), (-1, 0, 0, 0), "i*i must be -1")
    close(quat_multiply(k, k), (-1, 0, 0, 0), "k*k must be -1")
    q = (0.3, -0.2, 0.7, 0.5)
    close(quat_multiply(one, q), q, "identity*q must be q")
    close(quat_multiply(q, one), q, "q*identity must be q")


def test_quat_multiply_matches_reference_formula():
    rng = make_rng(11)
    for _ in range(50):
        a, b = random_quat(rng, unit=False), random_quat(rng, unit=False)
        out = quat_multiply(a, b)
        assert len(out) == 4, f"quat_multiply returned {out!r}, not a 4-tuple"
        close(out, ref_qmul(a, b), f"quat_multiply({a}, {b})")


def test_quat_from_axis_angle_values():
    same_rotation(quat_from_axis_angle((0, 0, 1), math.pi / 2), (H, 0, 0, H), "90 deg about z")
    close(ref_rotate(quat_from_axis_angle((0, 0, 1), math.pi / 2), (1, 0, 0)), (0, 1, 0), "90 deg about z sends x to y")
    close(ref_rotate(quat_from_axis_angle((1, 0, 0), math.pi / 2), (0, 1, 0)), (0, 0, 1), "90 deg about x sends y to z")
    close(ref_rotate(quat_from_axis_angle((0, 1, 0), math.pi / 2), (0, 0, 1)), (1, 0, 0), "90 deg about y sends z to x")
    close(ref_rotate(quat_from_axis_angle((0, 0, 1), -math.pi / 2), (1, 0, 0)), (0, -1, 0), "-90 deg about z sends x to -y")
    close(ref_rotate(quat_from_axis_angle((0, 0, 1), 1.5 * math.pi), (1, 0, 0)), (0, -1, 0), "270 deg about z sends x to -y")
    same_rotation(quat_from_axis_angle((0, 0, 2), 0.7), quat_from_axis_angle((0, 0, 1), 0.7), "non-unit axis must be normalized")
    rng = make_rng(12)
    for _ in range(20):
        axis, angle, v = random_vec(rng), rng.uniform(-4, 4), random_vec(rng)
        q = quat_from_axis_angle(axis, angle)
        same_rotation(q, q, f"axis {axis} angle {angle} must be a unit quaternion")
        close(ref_rotate(q, v), mat_vec(axis_angle_matrix(axis, angle), v), f"axis {axis} angle {angle} vs Rodrigues")


def test_quat_from_axis_angle_zero_axis_returns_identity():
    q = quat_from_axis_angle((0, 0, 0), 1.0)
    assert isinstance(q, tuple) and len(q) == 4, f"zero axis must give an identity quaternion, got {q!r}"
    close(q, IDENTITY_Q, "zero axis must give the identity quaternion")


def test_quat_from_rpy_is_fixed_axis_rz_ry_rx():
    close(ref_rotate(quat_from_rpy(math.pi / 2, 0, 0), (0, 1, 0)), (0, 0, 1), "roll 90 sends y to z")
    close(ref_rotate(quat_from_rpy(0, math.pi / 2, 0), (0, 0, 1)), (1, 0, 0), "pitch 90 sends z to x")
    close(ref_rotate(quat_from_rpy(0, 0, math.pi / 2), (1, 0, 0)), (0, 1, 0), "yaw 90 sends x to y")
    close(quat_from_rpy(0, 0, 0), IDENTITY_Q, "rpy (0,0,0) must be identity")
    rng = make_rng(13)
    for _ in range(25):
        r, p, y = (rng.uniform(-math.pi, math.pi) for _ in range(3))
        q = quat_from_rpy(r, p, y)
        same_rotation(q, q, f"rpy ({r}, {p}, {y}) must be a unit quaternion")
        close(quat_matrix_ref(q), ref_rpy_matrix(r, p, y), f"rpy ({r}, {p}, {y}) must equal Rz(y)*Ry(p)*Rx(r)")


def test_quat_to_rotation_matrix_and_rotate_vector_match_reference():
    rng = make_rng(14)
    for _ in range(25):
        q = random_quat(rng)
        R = quat_to_rotation_matrix(q)
        close(R, quat_matrix_ref(q), f"quat_to_rotation_matrix({q})")
        RRt = ref_matmul(R, [list(row) for row in zip(*R)])
        close(RRt, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], f"R*R^T must be identity for q={q}")
        close(quat_to_rotation_matrix(tuple(3 * c for c in q)), R, "non-unit quaternion must be normalized first")
        close(quat_to_rotation_matrix(tuple(-c for c in q)), R, "q and -q must give the same matrix")
        v = random_vec(rng)
        close(rotate_vector(q, v), ref_rotate(q, v), f"rotate_vector({q}, {v})")


def test_quat_normalize_unit_length_and_zero_is_identity():
    n = math.sqrt(30)
    close(quat_normalize((1, 2, 3, 4)), (1 / n, 2 / n, 3 / n, 4 / n), "normalize (1,2,3,4)")
    close(quat_normalize((0, 0, -5, 0)), (0, 0, -1, 0), "normalize keeps sign and direction")
    close(quat_normalize((H, 0, 0, H)), (H, 0, 0, H), "unit quaternion unchanged")
    close(quat_normalize((0, 0, 0, 0)), IDENTITY_Q, "zero quaternion must normalize to identity")


def test_mat4_from_translation_quat_layout():
    M = mat4_from_translation_quat((1, 2, 3), (H, 0, 0, H))
    close(M, [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], "Trans(1,2,3)*Rot(yaw 90)")
    rng = make_rng(15)
    for _ in range(10):
        q, t = random_quat(rng), random_vec(rng)
        close(mat4_from_translation_quat(t, q), homogeneous(quat_matrix_ref(q), t), f"mat4_from_translation_quat({t}, {q})")


def test_mat4_multiply_matches_naive_product():
    rng = make_rng(16)
    I = mat4_identity()
    close(I, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "mat4_identity")
    for _ in range(20):
        A = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(4)]
        B = [[rng.uniform(-3, 3) for _ in range(4)] for _ in range(4)]
        close(mat4_multiply(A, B), ref_matmul(A, B), "mat4_multiply(A, B) on general 4x4 matrices", tol=1e-8)
        close(mat4_multiply(I, A), A, "I*A must be A")
        close(mat4_multiply(A, I), A, "A*I must be A")
    T_a = mat4_from_translation_quat((1, 0, 0), IDENTITY_Q)
    T_b = mat4_from_translation_quat((0, 0, 0), (H, 0, 0, H))
    T_c = mat4_from_translation_quat((2, 0, 0), IDENTITY_Q)
    T = mat4_multiply(mat4_multiply(T_a, T_b), T_c)
    close([T[0][3], T[1][3], T[2][3]], (1, 2, 0), "move 1 on x, yaw 90, move 2 on new x lands at (1,2,0)")
    close(translation_of(mat4_multiply(T_b, T_a)), (0, 1, 0), "yaw 90 then move 1 on new x lands at (0,1,0)")
    close(translation_of(mat4_multiply(T_a, T_b)), (1, 0, 0), "move 1 on x then yaw 90 stays at (1,0,0)")


def test_column_major_and_ros_msg_adapters():
    M = [[4 * r + c for c in range(4)] for r in range(4)]
    out = mat4_to_column_major(M)
    assert out == [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], f"column-major order wrong: {out}"
    assert all(isinstance(x, float) for x in out), f"column-major entries must be floats: {out}"
    m = quat_to_msg((0.1, 0.2, 0.3, 0.4))
    assert m == {"x": 0.2, "y": 0.3, "z": 0.4, "w": 0.1}, f"quat_to_msg must put w last in the ROS dict: {m}"
    q = quat_from_msg({"x": 0.2, "y": 0.3, "z": 0.4, "w": 0.1})
    assert q == (0.1, 0.2, 0.3, 0.4), f"quat_from_msg must return (w, x, y, z): {q}"
    assert vec3_to_msg((1, 2, 3)) == {"x": 1.0, "y": 2.0, "z": 3.0}, vec3_to_msg((1, 2, 3))
    assert vec3_from_msg({"x": 1, "y": 2, "z": 3}) == (1.0, 2.0, 3.0), vec3_from_msg({"x": 1, "y": 2, "z": 3})
