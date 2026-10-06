"""servo.py — The joint servo law used by joint_state_publisher: one step of
moving a joint's position toward its setpoint. Pure functions, no
middleware, so tests/test_servo.py can check them directly.

The reference design (spec/FK_API.md, "joint_state_publisher") is a
proportional servo with gain 50 per second, clamped to the joint's limits.
"""

from __future__ import annotations

import math

from fk.urdf import Joint

GAIN_PER_S = 50.0


def clamp_to_limits(joint: Joint, value: float) -> float:
    """Clamp to `[lower, upper]` for revolute and prismatic joints. Continuous
    joints have no limits, so they are never clamped."""
    if joint.type in ("revolute", "prismatic"):
        return min(max(value, joint.lower), joint.upper)
    return value


def servo_step(joint: Joint, q: float, target: float, dt: float) -> float:
    """Return the joint's new position after `dt` seconds of moving from `q`
    toward `target`, with proportional gain `GAIN_PER_S`.

    Requirements (tests/test_servo.py):
      - converges to the target, never overshooting it, for any dt >= 0;
      - never leaves the joint's limits (revolute / prismatic), even when the
        target is outside them; continuous joints are never clamped;
      - dt == 0 leaves the joint where it is.
    """
    clamped_target = clamp_to_limits(joint, target)
    new_pos = q + (1 - math.exp(-GAIN_PER_S * dt)) * (clamped_target - q)
    return new_pos
