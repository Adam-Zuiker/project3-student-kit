"""URDF reading and validity. Run: python3 -m unittest discover -s tests -p test_urdf.py -v"""

from __future__ import annotations

import os
import unittest

from _helpers import read_robot
from fk.urdf import DescriptionError, find_root_and_children, parse_robot_description

FETCH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "robots", "fetch", "fetch.urdf")


def robot(body: str) -> str:
    return f'<robot name="r">{body}</robot>'


def revolute(name, parent, child, extra=""):
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<limit lower="-1" upper="1"/>{extra}</joint>')


class StructureTest(unittest.TestCase):
    """find_root_and_children on its own (validity rules 5 and 6)."""

    def _joints(self, *pairs):
        from fk.urdf import Joint
        return [Joint(name=f"j{i}", type="fixed", parent=p, child=c) for i, (p, c) in enumerate(pairs)]

    def test_chain(self):
        root, children = find_root_and_children(["a", "b", "c"], self._joints(("a", "b"), ("b", "c")))
        self.assertEqual(root, "a")
        self.assertEqual([j.child for j in children["a"]], ["b"])
        self.assertEqual(children["c"], [])

    def test_root_not_first(self):
        root, _ = find_root_and_children(["c", "b", "a"], self._joints(("a", "b"), ("a", "c")))
        self.assertEqual(root, "a")

    def test_single_link_no_joints(self):
        self.assertEqual(find_root_and_children(["only"], [])[0], "only")

    def test_unknown_link(self):
        with self.assertRaises(DescriptionError):
            find_root_and_children(["a", "b"], self._joints(("a", "zzz")))
        with self.assertRaises(DescriptionError):
            find_root_and_children(["a", "b"], self._joints(("zzz", "b")))

    def test_two_roots(self):
        with self.assertRaises(DescriptionError):
            find_root_and_children(["a", "b", "c"], self._joints(("a", "b")))

    def test_no_root(self):
        with self.assertRaises(DescriptionError):
            find_root_and_children(["a", "b"], self._joints(("a", "b"), ("b", "a")))


class AcceptTest(unittest.TestCase):
    def test_spec_example(self):
        m = parse_robot_description(read_robot("two_link_arm.urdf"))
        self.assertEqual(m.root, "base_link")
        self.assertEqual([j.name for j in m.joints], ["shoulder", "elbow_mount"])
        self.assertEqual(m.links["upper_arm"].visual.shape, "cylinder")

    def test_branching_root_last_and_hollow_joints_ignored(self):
        m = parse_robot_description(read_robot("branching_tree.urdf"))
        self.assertEqual(m.root, "torso")
        self.assertEqual(len(m.joints), 3)
        self.assertEqual(m.joints[1].axis, (0.0, 2.0, 0.0))  # kept as written

    def test_defaults(self):
        m = parse_robot_description(robot('<link name="a"/><link name="b"/>'
                                          '<joint name="j" type="continuous"><parent link="a"/><child link="b"/></joint>'))
        j = m.joints[0]
        self.assertEqual((j.origin.xyz, j.origin.rpy, j.axis), ((0, 0, 0), (0, 0, 0), (1, 0, 0)))

    def test_real_world_xml(self):
        text = ('\r\n  <?xml version="1.0" encoding="UTF-8"?>\r\n'
                '<robot xmlns="http://example.com/urdf" xmlns:xacro="http://www.ros.org/wiki/xacro">\r\n'
                "<!-- <joint name='fake'> -->\r\n"
                "<link name='a&amp;b'/><link name=\"c\"></link>\r\n"
                '<gazebo><![CDATA[ <joint name="nope"/> ]]></gazebo>\r\n'
                '<joint name="j" type="fixed"><parent link="a&#38;b"/><child link="c"/>'
                '<origin xyz=" 1\t.25\n3. " rpy="+1 1.5e-1 -2E+0"/><axis xyz="0 0 0"/></joint></robot>')
        m = parse_robot_description(text)
        self.assertEqual(m.root, "a&b")
        self.assertEqual(m.joints[0].origin.xyz, (1.0, 0.25, 3.0))

    @unittest.skipUnless(os.path.exists(FETCH), "kit's fetch.urdf not found")
    def test_fetch(self):
        with open(FETCH, encoding="utf-8") as f:
            m = parse_robot_description(f.read())
        self.assertEqual(m.root, "base_link")


class RejectTest(unittest.TestCase):
    CASES = {
        "not a string": 42,
        "empty": "   \n",
        "not xml": "<robot><link name='a'></robot>",
        "root not robot": '<model><link name="a"/></model>',
        "no links": "<robot/>",
        "nested link only": '<robot><gazebo><link name="a"/></gazebo></robot>',
        "link without name": "<robot><link/></robot>",
        "joint without type": robot('<link name="a"/><link name="b"/><joint name="j"><parent link="a"/><child link="b"/></joint>'),
        "joint without child": robot('<link name="a"/><joint name="j" type="fixed"><parent link="a"/></joint>'),
        "floating": robot('<link name="a"/><link name="b"/><joint name="j" type="floating"><parent link="a"/><child link="b"/></joint>'),
        "case variant": robot('<link name="a"/><link name="b"/><joint name="j" type="Revolute"><parent link="a"/><child link="b"/><limit/></joint>'),
        "unknown child": robot('<link name="a"/>' + revolute("j", "a", "ghost")),
        "two roots": robot('<link name="a"/><link name="b"/>'),
        "revolute no limit": robot('<link name="a"/><link name="b"/><joint name="j" type="revolute"><parent link="a"/><child link="b"/></joint>'),
        "bad limit": robot('<link name="a"/><link name="b"/>' + revolute("j", "a", "b").replace('lower="-1"', 'lower="low"')),
        "two numbers": robot('<link name="a"/><link name="b"/>' + revolute("j", "a", "b", '<origin xyz="1 2"/>')),
        "commas": robot('<link name="a"/><link name="b"/>' + revolute("j", "a", "b", '<axis xyz="0,0,1"/>')),
        "bad token": robot('<link name="a"/><link name="b"/>' + revolute("j", "a", "b", '<origin rpy="0 two 0"/>')),
        "bad fixed axis": robot('<link name="a"/><link name="b"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/><axis xyz="1 0"/></joint>'),
        "capsule": robot('<link name="a"><visual><geometry><capsule radius="1" length="1"/></geometry></visual></link>'),
        "empty geometry": robot('<link name="a"><visual><geometry/></visual></link>'),
        "box size count": robot('<link name="a"><visual><geometry><box size="1 1"/></geometry></visual></link>'),
        "mesh no filename": robot('<link name="a"><visual><geometry><mesh/></geometry></visual></link>'),
    }

    def test_each_rule(self):
        for label, text in self.CASES.items():
            with self.subTest(label):
                with self.assertRaises(DescriptionError):
                    parse_robot_description(text)


if __name__ == "__main__":
    unittest.main()
