"""fk/urdf.py — Turn `robot_description` text into a validated `RobotModel`,
or raise `DescriptionError` (spec/ROBOT_DESCRIPTION.md).

Pipeline:
  1. XML plumbing: strip a BOM / leading whitespace, parse, check the root
     is `<robot>`. Tags are compared by *local* name, because a default
     `xmlns="..."` on `<robot>` turns every tag into `{uri}link`.
  2. Per-element reading (`_read_link`, `_read_joint`): required
     attributes, joint types, strict numbers, defaults, visual geometry.
  3. Structure (`find_root_and_children`): every parent/child names a real
     link, exactly one root, and the parent -> joints map FK walks.

Every check runs before anything is returned, so a caller that only swaps
in a successfully returned model can never end up half-loaded.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from fk.transforms import Vec3

MOVABLE_TYPES = ("revolute", "continuous", "prismatic")
JOINT_TYPES = MOVABLE_TYPES + ("fixed",)
_LIMITED_TYPES = ("revolute", "prismatic")

# spec/ROBOT_DESCRIPTION.md, "Numbers"
_NUMBER_RE = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")
# XML whitespace only (str.split() would also split on e.g. \x0b or U+2003).
_XML_WS = " \t\r\n"
_XML_WS_RE = re.compile(r"[ \t\r\n]+")


class DescriptionError(ValueError):
    """The description must be rejected; `str(exc)` is the status `error`."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Origin:
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class Visual:
    """A link's first `<visual>`. Display-only; never affects FK."""

    shape: str  # "box" | "cylinder" | "sphere" | "mesh"
    params: Dict[str, object]
    origin: Origin = field(default_factory=Origin)


@dataclass
class Link:
    name: str
    visual: Optional[Visual] = None


@dataclass
class Joint:
    name: str
    type: str  # one of JOINT_TYPES
    parent: str
    child: str
    origin: Origin = field(default_factory=Origin)
    axis: Vec3 = (1.0, 0.0, 0.0)  # as written, NOT normalized
    lower: float = 0.0
    upper: float = 0.0

    @property
    def movable(self) -> bool:
        return self.type in MOVABLE_TYPES


@dataclass
class RobotModel:
    name: str
    links: Dict[str, Link]
    joints: List[Joint]  # document order
    root: str
    children: Dict[str, List[Joint]]  # parent link name -> joints leaving it


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def parse_number(text: Optional[str], what: str) -> float:
    token = (text or "").strip(_XML_WS)
    if not _NUMBER_RE.fullmatch(token):
        raise DescriptionError(f"{what}: {text!r} is not a number")
    value = float(token)
    if value in (float("inf"), float("-inf")):
        raise DescriptionError(f"{what}: {text!r} overflows")
    return value


def parse_vec3(text: Optional[str], what: str) -> Vec3:
    tokens = [t for t in _XML_WS_RE.split((text or "").strip(_XML_WS)) if t]
    if len(tokens) != 3:
        raise DescriptionError(f"{what}: expected 3 numbers, got {text!r}")
    x, y, z = (parse_number(t, what) for t in tokens)
    return (x, y, z)


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _local(tag: object) -> Optional[str]:
    """`{uri}link` -> `link`; comments/PIs (non-str tags) -> None."""
    if not isinstance(tag, str):
        return None
    return tag.rsplit("}", 1)[-1]


def _children(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    """Direct children only, by local name. Never `iter()`: nested
    `<joint>`s inside `<gazebo>`/`<transmission>` are not robot joints."""
    return (c for c in elem if _local(c.tag) == name)


def _first(elem: ET.Element, name: str) -> Optional[ET.Element]:
    return next(_children(elem, name), None)


def _read_origin(elem: Optional[ET.Element], what: str) -> Origin:
    if elem is None:
        return Origin()
    xyz = elem.get("xyz")
    rpy = elem.get("rpy")
    return Origin(
        xyz=parse_vec3(xyz, f"{what} origin xyz") if xyz is not None else (0.0, 0.0, 0.0),
        rpy=parse_vec3(rpy, f"{what} origin rpy") if rpy is not None else (0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Links and joints
# ---------------------------------------------------------------------------


def _read_visual(elem: ET.Element, link_name: str) -> Visual:
    what = f"link {link_name!r} visual"
    geometry = _first(elem, "geometry")
    shape_elem = next((c for c in geometry if _local(c.tag)), None) if geometry is not None else None
    if shape_elem is None:
        raise DescriptionError(f"{what}: missing or empty <geometry>")
    shape = _local(shape_elem.tag)
    params: Dict[str, object] = {}
    if shape == "box":
        params["size"] = parse_vec3(shape_elem.get("size"), f"{what} box size")
    elif shape == "cylinder":
        params["radius"] = parse_number(shape_elem.get("radius"), f"{what} cylinder radius")
        params["length"] = parse_number(shape_elem.get("length"), f"{what} cylinder length")
    elif shape == "sphere":
        params["radius"] = parse_number(shape_elem.get("radius"), f"{what} sphere radius")
    elif shape == "mesh":
        filename = shape_elem.get("filename")
        if filename is None:
            raise DescriptionError(f"{what}: mesh has no filename")
        params["filename"] = filename
        scale = shape_elem.get("scale")
        params["scale"] = parse_vec3(scale, f"{what} mesh scale") if scale is not None else (1.0, 1.0, 1.0)
    else:
        raise DescriptionError(f"{what}: unsupported geometry <{shape}>")
    return Visual(shape=shape, params=params, origin=_read_origin(_first(elem, "origin"), what))


def _read_link(elem: ET.Element) -> Link:
    name = elem.get("name")
    if name is None:
        raise DescriptionError("a <link> has no name")
    visual_elem = _first(elem, "visual")
    visual = _read_visual(visual_elem, name) if visual_elem is not None else None
    return Link(name=name, visual=visual)


def _read_joint(elem: ET.Element) -> Joint:
    name = elem.get("name")
    jtype = elem.get("type")
    if name is None or jtype is None:
        raise DescriptionError(f"a <joint> is missing name or type (name={name!r}, type={jtype!r})")
    what = f"joint {name!r}"
    if jtype not in JOINT_TYPES:
        raise DescriptionError(f"{what}: unsupported type {jtype!r}")

    ends = {}
    for end in ("parent", "child"):
        end_elem = _first(elem, end)
        link = end_elem.get("link") if end_elem is not None else None
        if link is None:
            raise DescriptionError(f"{what}: no <{end} link=...>")
        ends[end] = link

    axis_elem = _first(elem, "axis")
    axis_text = axis_elem.get("xyz") if axis_elem is not None else None
    # Parsed (so malformed is rejected) even for fixed joints, which ignore it.
    axis = parse_vec3(axis_text, f"{what} axis xyz") if axis_text is not None else (1.0, 0.0, 0.0)

    lower = upper = 0.0
    if jtype in _LIMITED_TYPES:
        limit = _first(elem, "limit")
        if limit is None:
            raise DescriptionError(f"{what}: {jtype} joint has no <limit>")
        if limit.get("lower") is not None:
            lower = parse_number(limit.get("lower"), f"{what} limit lower")
        if limit.get("upper") is not None:
            upper = parse_number(limit.get("upper"), f"{what} limit upper")

    return Joint(
        name=name,
        type=jtype,
        parent=ends["parent"],
        child=ends["child"],
        origin=_read_origin(_first(elem, "origin"), what),
        axis=axis,
        lower=lower,
        upper=upper,
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def find_root_and_children(
    link_names: List[str], joints: List[Joint]
) -> Tuple[str, Dict[str, List[Joint]]]:
    """Validate the robot's structure and return `(root, children)`, where
    `children[link]` lists the joints whose parent is `link` (every link
    gets a key, leaves map to `[]`).

    Raise `DescriptionError` when:
      - a joint's parent or child is not one of `link_names` (validity rule 5);
      - there is not exactly one root, i.e. one link that is no joint's
        child (validity rule 6).
    """
    tree = {}
    root_candidates = {}
    for link_name in link_names:
        tree[link_name] = []
        root_candidates[link_name] = True
    for joint in joints:
        if joint.child not in link_names or joint.parent not in link_names:
            raise DescriptionError("Either joint's child or parents is not in 'link names'")
        tree[joint.parent].append(joint)
        root_candidates[joint.child] = False

    count = sum(1 for value in root_candidates.values() if value is True)
    if count != 1:
        raise DescriptionError(f"Number of roots ({count}) != 1")
    else:
        root = [key for key, value in root_candidates.items() if value is True][0]

    return (root, tree)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_robot_description(value: object) -> RobotModel:
    if not isinstance(value, str):
        raise DescriptionError("robot_description is not a string")
    text = value.lstrip("﻿").lstrip(_XML_WS)
    if not text:
        raise DescriptionError("robot_description is empty")
    try:
        robot = ET.fromstring(text)
    except ET.ParseError as exc:
        raise DescriptionError(f"not well-formed XML: {exc}") from exc
    if _local(robot.tag) != "robot":
        raise DescriptionError(f"root element is <{_local(robot.tag)}>, not <robot>")

    links = [_read_link(e) for e in _children(robot, "link")]
    if not links:
        raise DescriptionError("<robot> has no <link>")
    joints = [_read_joint(e) for e in _children(robot, "joint")]

    root, children = find_root_and_children([l.name for l in links], joints)
    return RobotModel(
        name=robot.get("name", "robot"),
        links={l.name: l for l in links},
        joints=joints,
        root=root,
        children=children,
    )
