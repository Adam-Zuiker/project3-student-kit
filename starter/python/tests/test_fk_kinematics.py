"""Forward kinematics (src/fk/kinematics.py) and robot structure
(src/fk/urdf.py), checked against independent reference math."""

from __future__ import annotations

import itertools
import math

from _support import (
    BRANCHING_TREE, H, IDENTITY_Q, TWO_LINK_ARM, axis_angle_matrix, close,
    homogeneous, make_rng, quat_matrix_ref, random_rigid, random_vec,
    ref_matmul, ref_rotate, ref_rpy_matrix, run_with_timeout, same_rotation,
    translation_of,
)
from fk.kinematics import joint_transform, link_poses
from fk.transforms import mat4_from_translation_quat, mat4_identity
from fk.urdf import DescriptionError, Joint, Origin, find_root_and_children, parse_robot_description

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def joint_matrix(joint, q):
    t, rot = joint_transform(joint, q)
    assert len(t) == 3, f"{joint.name}: translation must have 3 entries, got {t!r}"
    return homogeneous(quat_matrix_ref(rot), t)


def test_joint_transform_revolute_continuous_fixed():
    j = Joint("j", "revolute", "a", "b", Origin(xyz=(1, 0, 0)), axis=(0, 0, 1))
    t, q = joint_transform(j, math.pi / 2)
    close(t, (1, 0, 0), "revolute motion must not move the joint origin")
    same_rotation(q, (H, 0, 0, H), "revolute 90 deg about z")
    same_rotation(joint_transform(j, 0.0)[1], IDENTITY_Q, "revolute at q=0 must not rotate")
    jc = Joint("j", "continuous", "a", "b", axis=(0, 0, 2))
    same_rotation(joint_transform(jc, math.pi / 2)[1], (H, 0, 0, H), "continuous axis (0,0,2) must be normalized")
    jf = Joint("j", "fixed", "a", "b", Origin(xyz=(1, 2, 3), rpy=(0.1, 0.2, 0.3)), axis=(0, 0, 1))
    t, q = joint_transform(jf, 5.0)
    close(t, (1, 2, 3), "fixed joint must ignore q (translation)")
    close(quat_matrix_ref(q), ref_rpy_matrix(0.1, 0.2, 0.3), "fixed joint must ignore q (rotation = origin rpy)")


def test_joint_transform_applies_origin_before_motion():
    j = Joint("j", "revolute", "a", "b", Origin(rpy=(math.pi / 2, 0, 0)), axis=(0, 0, 1))
    _, q = joint_transform(j, math.pi / 2)
    close(ref_rotate(q, (1, 0, 0)), (0, 0, 1), "T_origin*T_motion: motion sends x to y, then origin roll sends y to z")
    rng = make_rng(21)
    for jtype in ("revolute", "continuous"):
        for _ in range(10):
            xyz, rpy, axis, q = random_vec(rng), random_vec(rng, -3, 3), random_vec(rng), rng.uniform(-3, 3)
            j = Joint("j", jtype, "a", "b", Origin(xyz=xyz, rpy=rpy), axis=axis, lower=-9, upper=9)
            expected = ref_matmul(homogeneous(ref_rpy_matrix(*rpy), xyz), homogeneous(axis_angle_matrix(axis, q), (0, 0, 0)))
            close(joint_matrix(j, q), expected, f"{jtype} xyz={xyz} rpy={rpy} axis={axis} q={q}")


def test_joint_transform_prismatic_unnormalized_axis_rotated_by_origin():
    m = parse_robot_description(BRANCHING_TREE)
    slide = next(j for j in m.joints if j.name == "left_hand_slide")
    t, q = joint_transform(slide, 0.25)
    close(t, (0.3, 0.5, 0), "axis (2,0,0) at q=0.25 moves 0.5 along the joint x, which is the parent's y")
    same_rotation(q, (H, 0, 0, H), "prismatic must keep the origin rotation only")
    plain = Joint("p", "prismatic", "a", "b", axis=(2, 0, 0), lower=-5, upper=5)
    close(joint_transform(plain, 0.7)[0], (1.4, 0, 0), "spec: axis 2 0 0 at q=0.7 moves 1.4 m (not normalized)")
    close(joint_transform(plain, -0.5)[0], (-1.0, 0, 0), "prismatic negative q moves backwards")
    rng = make_rng(22)
    for _ in range(10):
        xyz, rpy, axis, q = random_vec(rng), random_vec(rng, -3, 3), random_vec(rng), rng.uniform(-2, 2)
        j = Joint("j", "prismatic", "a", "b", Origin(xyz=xyz, rpy=rpy), axis=axis, lower=-9, upper=9)
        expected = ref_matmul(homogeneous(ref_rpy_matrix(*rpy), xyz), homogeneous(I3, [q * a for a in axis]))
        close(joint_matrix(j, q), expected, f"prismatic xyz={xyz} rpy={rpy} axis={axis} q={q}")


def test_link_poses_chain_composes_parent_first_under_global_pose():
    rng = make_rng(23)
    A, B, C, G = (random_rigid(rng) for _ in range(4))
    edges = [("c", "d", C), ("a", "b", A), ("b", "c", B)]
    poses = link_poses(edges, G)
    assert set(poses) == {"a", "b", "c", "d"}, f"every link including the root must have a pose, got {sorted(poses)}"
    close(poses["a"], G, "root pose must be T_global_root")
    close(poses["b"], ref_matmul(G, A), "b = G*A")
    close(poses["c"], ref_matmul(ref_matmul(G, A), B), "c = G*A*B (parent first)")
    close(poses["d"], ref_matmul(ref_matmul(ref_matmul(G, A), B), C), "d = G*A*B*C (parent first)")


def test_link_poses_branching_tree_any_edge_order():
    rng = make_rng(24)
    L, R, Hd = (random_rigid(rng) for _ in range(3))
    base = [("torso", "left_arm", L), ("torso", "right_arm", R), ("left_arm", "left_hand", Hd)]
    for order in itertools.permutations(base):
        poses = link_poses(list(order), mat4_identity())
        names = [e[1] for e in order]
        assert set(poses) == {"torso", "left_arm", "right_arm", "left_hand"}, f"order {names}: got {sorted(poses)}"
        close(poses["torso"], mat4_identity(), f"order {names}: root torso must be identity")
        close(poses["right_arm"], R, f"order {names}: right_arm")
        close(poses["left_hand"], ref_matmul(L, Hd), f"order {names}: left_hand = left_arm*hand")


def test_link_poses_terminates_on_cycle_and_diamond():
    T = mat4_from_translation_quat((1, 0, 0), IDENTITY_Q)
    poses = run_with_timeout(lambda: link_poses([("a", "b", T), ("c", "d", T), ("d", "c", T)], mat4_identity()),
                             3, "link_poses with a detached cycle")
    assert "a" in poses and "b" in poses, f"root a and its child b must have poses, got {sorted(poses)}"
    close(translation_of(poses["b"]), (1, 0, 0), "b one step from the root")
    loop = [("a", "b", T), ("b", "c", T), ("c", "b", T)]
    poses = run_with_timeout(lambda: link_poses(loop, mat4_identity()), 3, "link_poses with a cycle reachable from the root")
    assert set(poses) == {"a", "b", "c"}, f"reachable cycle: got {sorted(poses)}"
    diamond = [("a", "b", T), ("a", "c", T), ("b", "d", T), ("c", "d", T)]
    poses = run_with_timeout(lambda: link_poses(diamond, mat4_identity()), 3, "link_poses with two paths to d")
    assert set(poses) == {"a", "b", "c", "d"}, f"diamond: got {sorted(poses)}"
    close(translation_of(poses["d"]), (2, 0, 0), "diamond: d two steps from the root")


def test_urdf_two_link_arm_fk_end_to_end():
    m = parse_robot_description(TWO_LINK_ARM)
    assert m.root == "base_link", f"root must be base_link, got {m.root!r}"
    shoulder = next(j for j in m.joints if j.name == "shoulder")
    assert (shoulder.lower, shoulder.upper) == (-3.14, 3.14), f"shoulder limits: {(shoulder.lower, shoulder.upper)}"
    close(shoulder.axis, (0, 0, 1), "shoulder axis")

    def fk(q_shoulder):
        qs = {"shoulder": q_shoulder}
        edges = [(j.parent, j.child, mat4_from_translation_quat(*joint_transform(j, qs.get(j.name, 0.0)))) for j in m.joints]
        return link_poses(edges, mat4_identity())

    poses = fk(0.0)
    assert set(poses) == {"base_link", "upper_arm", "forearm"}, f"links: {sorted(poses)}"
    close(translation_of(poses["upper_arm"]), (0, 0, 0.1), "q=0: upper_arm")
    close(translation_of(poses["forearm"]), (0.5, 0, 0.1), "q=0: forearm")
    close([r[2] for r in poses["forearm"][:3]], (1, 0, 0), "q=0: forearm z axis = world x (pitch 90)")
    poses = fk(math.pi / 2)
    close(translation_of(poses["forearm"]), (0, 0.5, 0.1), "shoulder 90: forearm swings to +y")
    close([r[2] for r in poses["forearm"][:3]], (0, 1, 0), "shoulder 90: forearm z axis = world y")


def test_find_root_and_children_structure_rules():
    m = parse_robot_description(BRANCHING_TREE)
    assert m.root == "torso", f"root must be torso (not the first link), got {m.root!r}"
    assert len(m.joints) == 3, f"nested <transmission><joint> is not a robot joint: {[j.name for j in m.joints]}"
    assert sorted(j.name for j in m.children["torso"]) == ["left_shoulder", "right_shoulder"], m.children["torso"]
    assert m.children["left_hand"] == [] and m.children["right_arm"] == [], "leaf links must map to []"

    def j(parent, child):
        return Joint(f"{parent}_{child}", "fixed", parent, child)

    root, children = find_root_and_children(["a", "b", "c"], [j("a", "b"), j("b", "c")])
    assert root == "a" and [x.child for x in children["a"]] == ["b"] and children["c"] == [], (root, children)
    for label, links, joints in (
        ("two roots", ["a", "b", "c"], [j("a", "b")]),
        ("no root (cycle)", ["a", "b"], [j("a", "b"), j("b", "a")]),
        ("unknown child link", ["a"], [j("a", "ghost")]),
        ("unknown parent link", ["a"], [j("ghost", "a")]),
    ):
        try:
            find_root_and_children(links, joints)
        except DescriptionError:
            continue
        raise AssertionError(f"find_root_and_children must raise DescriptionError for: {label}")
