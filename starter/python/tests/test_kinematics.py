"""Forward kinematics. Run: python3 -m unittest discover -s tests -p test_kinematics.py -v"""

from __future__ import annotations

import math
import unittest

from _helpers import H, assert_vec, read_robot, same_rotation, sandwich
from fk.kinematics import joint_transform, link_poses
from fk.transforms import QUAT_IDENTITY, mat4_from_translation_quat, mat4_identity
from fk.urdf import Joint, Origin, parse_robot_description


def translation(M):
    return [M[0][3], M[1][3], M[2][3]]


class JointTransformTest(unittest.TestCase):
    def test_revolute_pivots_about_its_origin(self):
        j = Joint("j", "revolute", "a", "b", Origin(xyz=(1, 0, 0)), axis=(0, 0, 1))
        t, q = joint_transform(j, math.pi / 2)
        assert_vec(self, t, (1, 0, 0))                # rotation does not move the joint origin
        same_rotation(self, q, (H, 0, 0, H))

    def test_non_unit_revolute_axis(self):
        j = Joint("j", "continuous", "a", "b", axis=(0, 0, 2))
        same_rotation(self, joint_transform(j, math.pi / 2)[1], (H, 0, 0, H))

    def test_origin_then_motion(self):
        # origin roll 90°, then rotate 90° about the joint's (rolled) z.
        j = Joint("j", "revolute", "a", "b", Origin(rpy=(math.pi / 2, 0, 0)), axis=(0, 0, 1))
        _, q = joint_transform(j, math.pi / 2)
        # T_origin·T_motion sends +x -> +y (motion) -> +z (origin roll).
        # The wrong order (motion·origin) would send +x to +y.
        assert_vec(self, sandwich(q, (1, 0, 0)), (0, 0, 1))

    def test_prismatic_axis_not_normalized_and_rotated_by_origin(self):
        m = parse_robot_description(read_robot("branching_tree.urdf"))
        slide = next(j for j in m.joints if j.name == "left_hand_slide")
        t, q = joint_transform(slide, 0.25)  # axis (2,0,0): moves 0.5 along the joint's x = parent's y
        assert_vec(self, t, (0.3, 0.5, 0))
        same_rotation(self, q, (H, 0, 0, H))

    def test_fixed_ignores_q(self):
        j = Joint("j", "fixed", "a", "b", Origin(xyz=(1, 2, 3)))
        t, q = joint_transform(j, 5.0)
        assert_vec(self, t, (1, 2, 3))
        same_rotation(self, q, QUAT_IDENTITY)


def _edges_at_zero(model):
    return [(j.parent, j.child, mat4_from_translation_quat(*joint_transform(j, 0.0))) for j in model.joints]


class LinkPosesTest(unittest.TestCase):
    def test_two_link_arm_zero_config(self):
        poses = link_poses(_edges_at_zero(parse_robot_description(read_robot("two_link_arm.urdf"))), mat4_identity())
        self.assertEqual(set(poses), {"base_link", "upper_arm", "forearm"})  # root included
        assert_vec(self, translation(poses["base_link"]), (0, 0, 0))
        assert_vec(self, translation(poses["forearm"]), (0.5, 0, 0.1))
        assert_vec(self, [r[2] for r in poses["forearm"][:3]], (1, 0, 0))  # pitch 90: forearm z = world x

    def test_global_pose_moves_everything(self):
        T_global_root = mat4_from_translation_quat((1, 2, 3), (H, 0, 0, H))  # yaw 90
        poses = link_poses(_edges_at_zero(parse_robot_description(read_robot("two_link_arm.urdf"))), T_global_root)
        assert_vec(self, translation(poses["base_link"]), (1, 2, 3))
        assert_vec(self, translation(poses["forearm"]), (1, 2.5, 3.1))

    def test_edge_order_does_not_matter(self):
        edges = list(reversed(_edges_at_zero(parse_robot_description(read_robot("branching_tree.urdf")))))
        poses = link_poses(edges, mat4_identity())
        self.assertEqual(set(poses), {"torso", "left_arm", "right_arm", "left_hand"})
        assert_vec(self, translation(poses["left_hand"]), (0.3, 0.2, 0.5))

    def test_terminates_on_cycle(self):
        T = mat4_identity()
        poses = link_poses([("a", "b", T), ("c", "d", T), ("d", "c", T)], mat4_identity())
        self.assertIn("a", poses)


if __name__ == "__main__":
    unittest.main()
