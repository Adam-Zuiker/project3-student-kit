"""Rotation math. Run: python3 -m unittest discover -s tests -p test_transforms.py -v"""

from __future__ import annotations

import math
import random
import unittest

from _helpers import H, assert_vec, conj, same_rotation, sandwich
from fk.transforms import (
    QUAT_IDENTITY,
    mat4_from_translation_quat,
    mat4_multiply,
    quat_from_axis_angle,
    quat_from_rpy,
    quat_multiply,
    quat_normalize,
    quat_to_rotation_matrix,
)


def _random_unit_quat(rng):
    return quat_normalize(tuple(rng.uniform(-1, 1) for _ in range(4)))


class QuatMultiplyTest(unittest.TestCase):
    def test_basis_products(self):
        i, j, k = (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        assert_vec(self, quat_multiply(i, j), k)                 # ij = k
        assert_vec(self, quat_multiply(j, i), (0, 0, 0, -1))     # ji = -k
        assert_vec(self, quat_multiply(i, i), (-1, 0, 0, 0))     # i² = -1

    def test_identity_and_inverse(self):
        rng = random.Random(1)
        for _ in range(20):
            q = _random_unit_quat(rng)
            assert_vec(self, quat_multiply(QUAT_IDENTITY, q), q)
            assert_vec(self, quat_multiply(q, conj(q)), QUAT_IDENTITY)

    def test_associative(self):
        rng = random.Random(2)
        a, b, c = (_random_unit_quat(rng) for _ in range(3))
        assert_vec(self, quat_multiply(quat_multiply(a, b), c), quat_multiply(a, quat_multiply(b, c)))


class AxisAngleTest(unittest.TestCase):
    def test_quarter_turn_about_z(self):
        same_rotation(self, quat_from_axis_angle((0, 0, 1), math.pi / 2), (H, 0, 0, H))
        assert_vec(self, sandwich(quat_from_axis_angle((0, 0, 1), math.pi / 2), (1, 0, 0)), (0, 1, 0))

    def test_non_unit_axis_is_normalized(self):
        same_rotation(self, quat_from_axis_angle((0, 0, 2), 0.7), quat_from_axis_angle((0, 0, 1), 0.7))
        q = quat_from_axis_angle((1, 1, 0), 1.3)
        self.assertAlmostEqual(sum(c * c for c in q), 1.0, delta=1e-12)

    def test_zero_axis_does_not_crash(self):
        quat_from_axis_angle((0, 0, 0), 1.0)

    def test_past_pi(self):
        # 3π/2 about z sends +x to -y
        assert_vec(self, sandwich(quat_from_axis_angle((0, 0, 1), 1.5 * math.pi), (1, 0, 0)), (0, -1, 0))


class RpyTest(unittest.TestCase):
    def test_single_axes(self):
        assert_vec(self, sandwich(quat_from_rpy(math.pi / 2, 0, 0), (0, 1, 0)), (0, 0, 1))  # roll: y -> z
        assert_vec(self, sandwich(quat_from_rpy(0, math.pi / 2, 0), (0, 0, 1)), (1, 0, 0))  # pitch: z -> x
        assert_vec(self, sandwich(quat_from_rpy(0, 0, math.pi / 2), (1, 0, 0)), (0, 1, 0))  # yaw: x -> y

    def test_fixed_axis_order(self):
        # R = Rz(yaw)·Ry(pitch)·Rx(roll): roll first, then pitch, then yaw, all about FIXED axes.
        # roll 90 sends y -> z; pitch 90 (about fixed y) sends z -> x; yaw 90 sends x -> y.
        assert_vec(self, sandwich(quat_from_rpy(math.pi / 2, math.pi / 2, math.pi / 2), (0, 1, 0)), (0, 1, 0))
        # and +x: roll leaves x; pitch sends x -> -z; yaw leaves -z.
        assert_vec(self, sandwich(quat_from_rpy(math.pi / 2, math.pi / 2, math.pi / 2), (1, 0, 0)), (0, 0, -1))

    def test_unit_norm(self):
        q = quat_from_rpy(0.3, -1.1, 2.9)
        self.assertAlmostEqual(sum(c * c for c in q), 1.0, delta=1e-12)


class RotationMatrixTest(unittest.TestCase):
    def test_matches_sandwich_product(self):
        rng = random.Random(3)
        for _ in range(20):
            q = _random_unit_quat(rng)
            R = quat_to_rotation_matrix(q)
            for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.3, -2.0, 1.5)):
                Rv = [sum(R[r][c] * v[c] for c in range(3)) for r in range(3)]
                assert_vec(self, Rv, sandwich(q, v))

    def test_q_and_minus_q_agree(self):
        q = quat_normalize((0.2, -0.5, 0.7, 0.1))
        mq = tuple(-c for c in q)
        for a, b in zip(quat_to_rotation_matrix(q), quat_to_rotation_matrix(mq)):
            assert_vec(self, a, b)


class Mat4Test(unittest.TestCase):
    def test_compose_translate_then_rotate(self):
        # T_a: move 1 along x. T_b: yaw 90° then move 2 along (the new) x.
        T_a = mat4_from_translation_quat((1, 0, 0), QUAT_IDENTITY)
        T_b = mat4_from_translation_quat((0, 0, 0), (H, 0, 0, H))
        T_c = mat4_from_translation_quat((2, 0, 0), QUAT_IDENTITY)
        T = mat4_multiply(mat4_multiply(T_a, T_b), T_c)
        assert_vec(self, [T[0][3], T[1][3], T[2][3]], (1, 2, 0))
        assert_vec(self, T[3], (0, 0, 0, 1))

    def test_not_commutative(self):
        A = mat4_from_translation_quat((1, 0, 0), QUAT_IDENTITY)
        B = mat4_from_translation_quat((0, 0, 0), (H, 0, 0, H))
        self.assertNotAlmostEqual(mat4_multiply(A, B)[1][3], mat4_multiply(B, A)[1][3])


if __name__ == "__main__":
    unittest.main()
