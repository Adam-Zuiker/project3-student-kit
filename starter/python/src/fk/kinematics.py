"""fk/kinematics.py — Forward kinematics.

Two halves, matching the two graded nodes:

  - `joint_transform`  (robot_state_publisher): one joint's
    `T_joint(q) = T_origin · T_motion(q)`, as translation + quaternion,
    which is exactly what one `/tf` entry carries.
  - `link_poses`       (robot_world_state_publisher): given the edges of
    one `/tf` message as 4x4 matrices, find the root and compose down the
    tree into every link's pose in the global frame.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from fk.transforms import Mat4, Quat, Vec3, quat_from_rpy, quat_from_axis_angle, rotate_vector, quat_multiply, QUAT_IDENTITY, add, scale, mat4_identity, mat4_multiply

from fk.urdf import Joint, find_root_and_children

# One /tf entry, already converted to a matrix: (parent, child, T_parent_child)
Edge = Tuple[str, str, Mat4]


def joint_transform(joint: Joint, q: float) -> Tuple[Vec3, Quat]:
    """`T_origin · T_motion(q)` for one joint, as `(translation, rotation)`.

    - `T_origin = Trans(origin.xyz) · Rot(origin.rpy)`
    - revolute / continuous: rotate by `q` about the *normalized* axis
    - prismatic: translate by `q · axis`, axis *as written*
    - fixed: identity motion
    The motion is expressed in the joint frame, i.e. *after* the origin.
    """
    t_o = joint.origin.xyz
    r_o = quat_from_rpy(*joint.origin.rpy)

    if joint.type == "revolute" or joint.type == "continuous":
      t_m = (0,0,0)
      r_m = quat_from_axis_angle(joint.axis, q)

    elif joint.type == "prismatic":
       t_m = tuple(scale(joint.axis, q))
       r_m = QUAT_IDENTITY

    else:
       t_m = (0,0,0)
       r_m = QUAT_IDENTITY

    translation = tuple(add(list(t_o), list(rotate_vector(r_o, t_m))))
    rotation = quat_multiply(r_o, r_m)

    return(translation, rotation)



def link_poses(edges: List[Edge], T_global_root: Mat4) -> Dict[str, Mat4]:
    """Every link's pose in the global frame, root included.

    The root is the frame that is some edge's parent and no edge's child.
    `T_global_link = T_global_root · T_root_link`, where `T_root_link` is the
    product of the edge matrices along the path from the root, parent first.
    Must terminate even if the edges are not a tree.
    """
    tree = {}
    children = set()
    for edge in edges:
       if edge[0] in tree:
          tree[edge[0]].append((edge[1], edge[2])) 
       else:
          tree[edge[0]] = [(edge[1], edge[2])]
       children.add(edge[1])

    # root finding
    root = None
    for parent in tree.keys():
       if parent not in children:
          root = parent
          break

    if root is None:
       raise ValueError("Invalid tree: no root found")

    # traverse through tree from root using DFS so want to use stack
    poses = {}
    tf_stack = [(root, mat4_multiply(mat4_identity(), T_global_root))]

    while len(tf_stack) > 0:
       curr_link, transform = tf_stack.pop()
       poses[curr_link] = transform
       curr_children = tree.get(curr_link, [])
       for child in curr_children:
          if child[0] not in poses:
            tf_stack.append((child[0], mat4_multiply(transform, child[1])))

    return poses
  