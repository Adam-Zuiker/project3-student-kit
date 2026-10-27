"""FSM validation and arrival (src/fsm_logic.py) and the state machine
itself (src/finite_state_machine.py), driven directly with no runtime."""

from __future__ import annotations

import glob
import json
import os

from _support import FakeClock, PROJECT_DIR, joint_states, make_fsm, messages_on, spec_fsm

import finite_state_machine
from fsm_logic import arrived, validate_fsm

JOINTS = ["shoulder", "elbow"]


def edited(fn):
    doc = spec_fsm()
    fn(doc)
    return doc


def last_target(published):
    trajectories = messages_on(published, "/joint_trajectory")
    assert trajectories, "no /joint_trajectory was published"
    t = trajectories[-1]
    return t["joint_names"], t["points"][0]["positions"]


def last_status(published):
    statuses = messages_on(published, "/fsm/status")
    assert statuses, "no /fsm/status was published"
    return statuses[-1]


def test_validate_fsm_accepts_valid_tables():
    cases = {
        "spec example": spec_fsm(),
        "zero epsilon and no on_arrival": edited(lambda d: (d["states"][1].pop("on_arrival"), d["states"][0].update(epsilon=0))),
        "on_arrival null": edited(lambda d: d["states"][1].update(on_arrival=None)),
        "integer positions and epsilon": edited(lambda d: (d["states"][0]["target"].update(positions=[0, 1]), d["states"][0].update(epsilon=1))),
        "same transition name in two states": edited(lambda d: d["states"][1]["transitions"].append({"name": "to_wave", "next_state": "wave"})),
        "terminal state without transitions": edited(lambda d: (d["states"][1].pop("on_arrival"), d["states"][1].update(transitions=[]))),
        "hold extension": edited(lambda d: d["states"][0].update(hold=0.5)),
        "self loop": edited(lambda d: d["states"][0]["transitions"].append({"name": "stay", "next_state": "home"})),
        "single state": {"initial_state": "only", "states": [
            {"name": "only", "target": {"joint_names": [], "positions": []}, "epsilon": 0, "transitions": []}]},
    }
    for path in sorted(glob.glob(os.path.join(PROJECT_DIR, "fsm", "*.json"))):
        with open(path, encoding="utf-8") as f:
            cases[f"shipped fsm/{os.path.basename(path)}"] = json.load(f)
    for label, doc in cases.items():
        reason = validate_fsm(doc)
        assert reason is None, f"valid table rejected ({label}): {reason}"


def test_validate_fsm_rejects_every_rule():
    cases = {
        "rule 1: no states": edited(lambda d: d.update(states=[])),
        "rule 2: duplicate state names": edited(lambda d: d["states"].append(spec_fsm()["states"][0])),
        "rule 3: duplicate transition names in one state": edited(
            lambda d: d["states"][0]["transitions"].append({"name": "to_wave", "next_state": "home"})),
        "rule 4: next_state does not resolve": edited(lambda d: d["states"][0]["transitions"][0].update(next_state="nowhere")),
        "rule 5: on_arrival names no transition": edited(lambda d: d["states"][0].update(on_arrival="to_moon")),
        "rule 5: on_arrival names another state's transition": edited(lambda d: d["states"][0].update(on_arrival="to_home")),
        "rule 5: on_arrival is a list": edited(lambda d: d["states"][0].update(on_arrival=["to_wave"])),
        "rule 6: initial_state does not resolve": edited(lambda d: d.update(initial_state="nowhere")),
        "rule 6: missing initial_state": edited(lambda d: d.pop("initial_state")),
        "rule 6: initial_state null": edited(lambda d: d.update(initial_state=None)),
        "rule 7: negative epsilon": edited(lambda d: d["states"][1].update(epsilon=-0.1)),
        "shape: string epsilon": edited(lambda d: d["states"][0].update(epsilon="0.02")),
        "shape: boolean epsilon": edited(lambda d: d["states"][0].update(epsilon=True)),
        "shape: not an object": ["home"],
        "shape: null": None,
        "shape: states not a list": edited(lambda d: d.update(states={"home": {}})),
        "shape: state not an object": edited(lambda d: d["states"].append("wave")),
        "shape: state without name": edited(lambda d: d["states"][0].pop("name")),
        "shape: target lengths differ": edited(lambda d: d["states"][0]["target"].update(positions=[0.0])),
        "shape: non-numeric position": edited(lambda d: d["states"][0]["target"].update(positions=[0.0, "up"])),
        "shape: boolean position": edited(lambda d: d["states"][0]["target"].update(positions=[0.0, False])),
        "shape: non-string joint name": edited(lambda d: d["states"][0]["target"].update(joint_names=["shoulder", 7])),
        "shape: missing target": edited(lambda d: d["states"][0].pop("target")),
        "shape: transitions not a list": edited(lambda d: d["states"][0].update(transitions="to_wave")),
        "shape: transition without next_state": edited(lambda d: d["states"][0]["transitions"][0].pop("next_state")),
        "hold: negative": edited(lambda d: d["states"][0].update(hold=-1)),
        "hold: string": edited(lambda d: d["states"][0].update(hold="1s")),
    }
    for label, doc in cases.items():
        try:
            reason = validate_fsm(doc)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"validate_fsm raised {exc!r} instead of returning a reason ({label})") from exc
        assert isinstance(reason, str) and reason, f"invalid table accepted ({label}): returned {reason!r}"


def test_arrived_uses_epsilon_inclusively_and_needs_every_joint():
    s = {"name": "s", "target": {"joint_names": ["a", "b"], "positions": [1.0, -1.0]}, "epsilon": 0.05, "transitions": []}
    assert arrived(s, {"a": 1.04, "b": -0.96}), "both joints within epsilon must count as arrived"
    assert not arrived(s, {"a": 1.0, "b": -0.9}), "b is 0.1 away (> 0.05): not arrived"
    assert not arrived(s, {"a": 0.9, "b": -1.0}), "a is 0.1 below target: not arrived (needs abs)"
    assert not arrived(s, {"a": 1.1, "b": -1.0}), "a is 0.1 above target: not arrived"
    assert not arrived(s, {"a": -1.0, "b": 1.0}), "positions swapped between joints: not arrived"
    assert not arrived(s, {"a": 1.0}), "missing joint b: not arrived"
    assert arrived(s, {"a": 1.0, "b": -1.0, "c": 99.0}), "joints not in the target must be ignored"
    wide = dict(s, epsilon=0.5)
    assert arrived(wide, {"a": 1.5, "b": -1.5}), "exactly epsilon away (above and below) counts as arrived"
    exact = dict(s, epsilon=0)
    assert arrived(exact, {"a": 1.0, "b": -1.0}), "epsilon 0 with exact positions: arrived"
    assert not arrived(exact, {"a": 1.0, "b": -1.0001}), "epsilon 0 with any error: not arrived"
    empty = dict(s, target={"joint_names": [], "positions": []})
    assert arrived(empty, {}), "an empty target has always arrived"


def test_fsm_runtime_follows_on_arrival_after_reaching_target():
    fsm, published = make_fsm()
    values, ok, status = fsm._update_logic(spec_fsm())
    assert ok and values == {"current_state": "home"}, (values, ok, status)
    assert last_target(published) == (JOINTS, [0.0, 0.0]), f"entering home must publish its target: {last_target(published)}"

    fsm._tick()
    st = last_status(published)
    assert st["current_state"] == "home" and st["running"] is True and st["last_transition"] is None, st

    fsm._on_joint_states(joint_states(JOINTS, [0.5, 0.0]))
    fsm._tick()
    assert last_status(published)["current_state"] == "home", "far from target: must stay in home"

    fsm._on_joint_states(joint_states(JOINTS, [True, float("nan")]))
    fsm._tick()
    assert last_status(published)["current_state"] == "home", "bool/NaN joint positions must be ignored, not arrived"

    fsm._on_joint_states(joint_states(JOINTS, [0.01, -0.01]))
    fsm._tick()
    st = last_status(published)
    assert st["current_state"] == "wave" and st["last_transition"] == "to_wave", f"within epsilon of home: {st}"
    assert last_target(published) == (JOINTS, [0.8, -0.5]), f"entering wave must publish its target: {last_target(published)}"

    fsm._on_joint_states(joint_states(JOINTS, [0.8, -0.5]))
    fsm._tick()
    st = last_status(published)
    assert st["current_state"] == "home" and st["last_transition"] == "to_home", f"at wave target: {st}"


def test_fsm_runtime_fire_transition_set_running_and_get_logic():
    fsm, published = make_fsm()
    assert fsm._fire_transition({"transition": "to_wave"})[1] is False, "fire before any FSM is loaded must fail"
    assert fsm._get_logic({})[1] is False, "get_logic before any FSM is loaded must fail"

    doc = spec_fsm()
    fsm._update_logic(doc)
    got, ok, _ = fsm._get_logic({})
    assert ok and got == {"initial_state": "home", "states": spec_fsm()["states"]}, got
    got["states"].clear()
    assert fsm._get_logic({})[0]["states"], "get_logic must return a copy, not the live table"

    n = len(messages_on(published, "/joint_trajectory"))
    values, ok, _ = fsm._fire_transition({"transition": "to_moon"})
    assert ok is False and fsm._current == "home", "unknown transition must fail and leave the state alone"
    assert len(messages_on(published, "/joint_trajectory")) == n, "a failed fire must not publish a trajectory"
    assert fsm._fire_transition({"transition": 3})[1] is False, "non-string transition must fail"

    values, ok, _ = fsm._fire_transition({"transition": "to_wave"})
    assert ok and values == {"current_state": "wave"}, values
    assert last_target(published) == (JOINTS, [0.8, -0.5]), last_target(published)

    assert fsm._set_running({"data": False}) == ({"running": False}, True, ""), "pause must report running False"
    fsm._on_joint_states(joint_states(JOINTS, [0.8, -0.5]))
    fsm._tick()
    assert fsm._current == "wave", "a paused FSM must not follow on_arrival"
    assert last_status(published)["running"] is False, last_status(published)
    assert fsm._fire_transition({"transition": "to_home"})[1] is False, "fire while paused must fail"
    assert fsm._set_running({"data": "yes"})[1] is False, "non-boolean data must fail"
    assert fsm._set_running({})[0] == {"running": False}, "no data must just report the current state"

    n = len(messages_on(published, "/joint_trajectory"))
    assert fsm._set_running({"data": True})[0] == {"running": True}
    assert len(messages_on(published, "/joint_trajectory")) == n + 1, "resuming must re-send the current target"
    fsm._tick()
    assert fsm._current == "home", "after resuming at the wave target, on_arrival must fire"

    values, ok, _ = fsm._update_logic({"initial_state": "x", "states": []})
    assert ok is False and fsm._current == "home", "an invalid table must be refused and keep the old one running"


def test_fsm_runtime_hold_waits_before_on_arrival():
    clock = FakeClock().install(finite_state_machine)
    fsm, published = make_fsm()
    fsm._update_logic(edited(lambda d: d["states"][0].update(hold=2.0)))
    at_home = joint_states(JOINTS, [0.0, 0.0])
    away = joint_states(JOINTS, [0.5, 0.0])

    fsm._on_joint_states(away)
    fsm._tick()
    assert fsm._current == "home", "not at target yet: must stay in home"

    fsm._on_joint_states(at_home)
    fsm._tick()
    assert fsm._current == "home", "just arrived with hold 2 s: must not transition on the first tick"
    clock.advance(1.5)
    fsm._tick()
    assert fsm._current == "home", "1.5 s into a 2 s hold: must stay in home"

    fsm._on_joint_states(away)
    fsm._tick()
    fsm._on_joint_states(at_home)
    fsm._tick()
    clock.advance(1.75)
    fsm._tick()
    assert fsm._current == "home", "left the target at 1.5 s and came back 1.75 s ago: hold must restart"

    clock.advance(0.25)
    fsm._tick()
    st = last_status(published)
    assert fsm._current == "wave", "exactly 2 s at the target since returning: hold is over, must follow on_arrival"
    assert st["current_state"] == "wave" and st["last_transition"] == "to_wave", st
    assert last_target(published) == (JOINTS, [0.8, -0.5]), f"entering wave must publish its target: {last_target(published)}"
