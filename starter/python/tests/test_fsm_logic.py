"""FSM table validation and arrival. Run: python3 -m unittest discover -s tests -p test_fsm_logic.py -v"""

from __future__ import annotations

import copy
import json
import os
import unittest

import _helpers  # noqa: F401 - puts src/ on the path
from fsm_logic import arrived, validate_fsm

# The spec's worked example.
SPEC_EXAMPLE = {
    "initial_state": "home",
    "states": [
        {"name": "home",
         "target": {"joint_names": ["shoulder", "elbow"], "positions": [0.0, 0.0]},
         "epsilon": 0.02,
         "on_arrival": "to_wave",
         "transitions": [{"name": "to_wave", "next_state": "wave"}]},
        {"name": "wave",
         "target": {"joint_names": ["shoulder", "elbow"], "positions": [0.8, -0.5]},
         "epsilon": 0.02,
         "on_arrival": "to_home",
         "transitions": [{"name": "to_home", "next_state": "home"}]},
    ],
}
FETCH_WAVE = os.path.join(os.path.dirname(__file__), "..", "fsm", "fetch_wave.json")


def mutated(fn):
    doc = copy.deepcopy(SPEC_EXAMPLE)
    fn(doc)
    return doc


class ValidateAcceptTest(unittest.TestCase):
    def test_spec_example(self):
        self.assertIsNone(validate_fsm(SPEC_EXAMPLE))

    def test_shipped_fetch_choreography(self):
        with open(FETCH_WAVE, encoding="utf-8") as f:
            self.assertIsNone(validate_fsm(json.load(f)))

    def test_on_arrival_optional_and_zero_epsilon(self):
        doc = mutated(lambda d: (d["states"][1].pop("on_arrival"), d["states"][0].update(epsilon=0)))
        self.assertIsNone(validate_fsm(doc))

    def test_same_transition_name_in_different_states(self):
        doc = mutated(lambda d: d["states"][1]["transitions"].append({"name": "to_wave", "next_state": "wave"}))
        self.assertIsNone(validate_fsm(doc))

    def test_terminal_state_without_transitions(self):
        doc = mutated(lambda d: (d["states"][1].pop("on_arrival"), d["states"][1].update(transitions=[])))
        self.assertIsNone(validate_fsm(doc))


class ValidateRejectTest(unittest.TestCase):
    CASES = {
        # spec list
        "no states": mutated(lambda d: d.update(states=[])),
        "duplicate state names": mutated(lambda d: d["states"][1].update(name="home")),
        "duplicate transition names in one state": mutated(
            lambda d: d["states"][0]["transitions"].append({"name": "to_wave", "next_state": "home"})),
        "next_state does not resolve": mutated(lambda d: d["states"][0]["transitions"][0].update(next_state="nowhere")),
        "on_arrival names no transition": mutated(lambda d: d["states"][0].update(on_arrival="to_moon")),
        "on_arrival names another state's transition": mutated(lambda d: d["states"][0].update(on_arrival="to_home")),
        "initial_state does not resolve": mutated(lambda d: d.update(initial_state="nowhere")),
        "missing initial_state": mutated(lambda d: d.pop("initial_state")),
        "negative epsilon": mutated(lambda d: d["states"][0].update(epsilon=-0.1)),
        "string epsilon": mutated(lambda d: d["states"][0].update(epsilon="0.02")),
        "boolean epsilon": mutated(lambda d: d["states"][0].update(epsilon=True)),
        # shape
        "not an object": ["home"],
        "null": None,
        "states not a list": mutated(lambda d: d.update(states={"home": {}})),
        "state not an object": mutated(lambda d: d["states"].append("wave")),
        "state without name": mutated(lambda d: d["states"][0].pop("name")),
        "target lengths differ": mutated(lambda d: d["states"][0]["target"].update(positions=[0.0])),
        "non-numeric position": mutated(lambda d: d["states"][0]["target"].update(positions=[0.0, "up"])),
        "missing target": mutated(lambda d: d["states"][0].pop("target")),
        "transitions not a list": mutated(lambda d: d["states"][0].update(transitions="to_wave")),
        "transition without next_state": mutated(lambda d: d["states"][0]["transitions"][0].pop("next_state")),
    }

    def test_each_rule(self):
        for label, doc in self.CASES.items():
            with self.subTest(label):
                try:
                    reason = validate_fsm(doc)
                except Exception as exc:  # noqa: BLE001
                    self.fail(f"raised {exc!r} instead of returning a reason")
                self.assertIsInstance(reason, str)
                self.assertTrue(reason, "the reason must be non-empty")


class ArrivedTest(unittest.TestCase):
    STATE = {"name": "s", "target": {"joint_names": ["a", "b"], "positions": [1.0, -1.0]},
             "epsilon": 0.05, "transitions": []}

    def test_all_within_epsilon(self):
        self.assertTrue(arrived(self.STATE, {"a": 1.04, "b": -0.96}))

    def test_one_joint_outside(self):
        self.assertFalse(arrived(self.STATE, {"a": 1.0, "b": -0.9}))

    def test_boundary_counts_as_arrived(self):
        state = dict(self.STATE, epsilon=0.5)
        self.assertTrue(arrived(state, {"a": 1.5, "b": -1.0}))

    def test_missing_joint_has_not_arrived(self):
        self.assertFalse(arrived(self.STATE, {"a": 1.0}))

    def test_extra_joints_ignored(self):
        self.assertTrue(arrived(self.STATE, {"a": 1.0, "b": -1.0, "c": 99.0}))

    def test_zero_epsilon_needs_exact(self):
        state = dict(self.STATE, epsilon=0)
        self.assertTrue(arrived(state, {"a": 1.0, "b": -1.0}))
        self.assertFalse(arrived(state, {"a": 1.0, "b": -1.0001}))

    def test_empty_target_has_arrived(self):
        state = dict(self.STATE, target={"joint_names": [], "positions": []})
        self.assertTrue(arrived(state, {}))


if __name__ == "__main__":
    unittest.main()
