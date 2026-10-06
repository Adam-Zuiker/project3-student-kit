"""Joint servo law. Run: python3 -m unittest discover -s tests -p test_servo.py -v"""

from __future__ import annotations

import unittest

import _helpers  # noqa: F401 - puts src/ on the path
from fk.urdf import Joint
from servo import servo_step

REVOLUTE = Joint("r", "revolute", "a", "b", lower=-1.0, upper=1.0)
CONTINUOUS = Joint("c", "continuous", "a", "b")
PRISMATIC = Joint("p", "prismatic", "a", "b", lower=0.0, upper=0.3)


def run(joint, q, target, dt, steps):
    trace = [q]
    for _ in range(steps):
        q = servo_step(joint, q, target, dt)
        trace.append(q)
    return trace


class ServoTest(unittest.TestCase):
    def test_converges_at_demo_rate(self):
        trace = run(REVOLUTE, -0.8, 0.5, 0.1, 20)  # 2 s at 10 Hz
        self.assertAlmostEqual(trace[-1], 0.5, delta=1e-3)

    def test_never_overshoots_and_error_never_grows(self):
        for dt in (0.001, 0.01, 0.02, 0.1, 0.5):  # 0.1 s · 50/s = 5: a naive Euler P-step explodes here
            with self.subTest(dt=dt):
                trace = run(CONTINUOUS, 0.0, 2.0, dt, 200)
                errors = [2.0 - q for q in trace]
                self.assertTrue(all(e >= -1e-12 for e in errors), "overshot the target")
                self.assertTrue(all(b <= a + 1e-12 for a, b in zip(errors, errors[1:])), "error grew")

    def test_moves_toward_target_each_step(self):
        q = servo_step(CONTINUOUS, 1.0, -1.0, 0.01)
        self.assertLess(q, 1.0)
        self.assertGreaterEqual(q, -1.0)

    def test_dt_zero_holds_position(self):
        self.assertEqual(servo_step(REVOLUTE, 0.25, 0.9, 0.0), 0.25)

    def test_target_outside_limits_stops_at_limit(self):
        trace = run(REVOLUTE, 0.0, 5.0, 0.1, 30)
        self.assertTrue(all(-1.0 <= q <= 1.0 for q in trace))
        self.assertAlmostEqual(trace[-1], 1.0, delta=1e-3)
        trace = run(PRISMATIC, 0.1, -2.0, 0.1, 30)
        self.assertTrue(all(0.0 <= q <= 0.3 for q in trace))
        self.assertAlmostEqual(trace[-1], 0.0, delta=1e-3)

    def test_continuous_is_never_clamped(self):
        self.assertAlmostEqual(run(CONTINUOUS, 0.0, 10.0, 0.1, 40)[-1], 10.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
