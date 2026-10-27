"""fsm_logic.py — The decision-making core of finite_state_machine, as pure
functions on plain JSON data (no middleware), so tests/test_fsm_logic.py can
check them directly.

An FSM document (spec/FK_API.md, "finite_state_machine"):

    {"initial_state": "home",
     "states": [
       {"name": "home",
        "target": {"joint_names": ["shoulder", "elbow"], "positions": [0.0, 0.0]},
        "epsilon": 0.02,
        "on_arrival": "to_wave",                       # optional
        "transitions": [{"name": "to_wave", "next_state": "wave"}]},
       ...]}

The document arrives straight off the wire, so it can be *any* JSON value.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _shape_error(doc: Any) -> Optional[str]:
    """Rule 8, the JSON shape. After this returns None, every field below
    exists with the right type, so validate_fsm can index freely:

        doc["initial_state"]                       (any value; checked by rule 6)
        doc["states"]                              list of dicts
        state["name"]                              str
        state["target"]["joint_names"]             list of str
        state["target"]["positions"]               list of numbers, same length
        state["epsilon"]                           number
        state["transitions"]                       list of dicts
        transition["name"], transition["next_state"]   str
        state.get("on_arrival")                    absent/None, or any value (rule 5)
    """
    if not isinstance(doc, dict):
        return "the FSM document must be an object"
    states = doc.get("states")
    if not isinstance(states, list):
        return "states must be a list"
    for i, s in enumerate(states):
        if not isinstance(s, dict) or not isinstance(s.get("name"), str):
            return f"states[{i}] must be an object with a string name"
        target = s.get("target")
        if (not isinstance(target, dict)
                or not isinstance(target.get("joint_names"), list)
                or not isinstance(target.get("positions"), list)
                or len(target["joint_names"]) != len(target["positions"])
                or not all(isinstance(n, str) for n in target["joint_names"])
                or not all(_is_number(p) for p in target["positions"])):
            return f"state {s['name']!r}: target needs equal-length joint_names (strings) and positions (numbers)"
        if not _is_number(s.get("epsilon")):
            return f"state {s['name']!r}: epsilon must be a number"
        # Optional extension (not in the spec): seconds to stay after arriving.
        if "hold" in s and not (_is_number(s["hold"]) and s["hold"] >= 0):
            return f"state {s['name']!r}: hold must be a number of seconds >= 0"
        transitions = s.get("transitions")
        if not isinstance(transitions, list) or not all(
                isinstance(t, dict) and isinstance(t.get("name"), str) and isinstance(t.get("next_state"), str)
                for t in transitions):
            return f"state {s['name']!r}: transitions must be objects with string name and next_state"
    return None


def validate_fsm(doc: Any) -> Optional[str]:
    """Return None if `doc` is a valid FSM table, else a short reason.

    Valid means (spec/FK_API.md, "finite_state_machine"):
      1. at least one state;
      2. state names are unique;
      3. transition names are unique *within* each state;
      4. every transition's `next_state` names a state;
      5. `on_arrival`, if present, names one of that state's own transitions;
      6. `initial_state` names a state;
      7. every `epsilon` is >= 0;
      8. the JSON shape (`_shape_error`, already written).
    Must never raise, whatever `doc` is.
    """
    shape = _shape_error(doc)
    if shape is not None:
        return shape

    # Rule 1
    if len(doc["states"]) == 0:
        return "the FSM has no states (need at least one)"

    # Rule 2
    state_names_set = set()
    state_names_list = []
    for state in doc["states"]:
        state_names_set.add(state["name"])
        state_names_list.append(state["name"])
    if len(state_names_set) < len(state_names_list):
        duplicate = next(n for n in state_names_list if state_names_list.count(n) > 1)
        return f"state name {duplicate!r} is used by more than one state"

    # Rules 3, 4, 5, 7
    for state in doc["states"]:
        if state['epsilon'] < 0:
            return f"state {state['name']!r}: epsilon is {state['epsilon']}, must be >= 0"
        
        transitions_set = set()
        transitions_list = []
        for transition in state['transitions']:
            transitions_set.add(transition["name"])
            transitions_list.append(transition["name"])
            if transition["next_state"] not in state_names_set:
                return (f"state {state['name']!r}: transition {transition['name']!r} goes to "
                        f"{transition['next_state']!r}, which is not a state")
        if len(transitions_set) < len(transitions_list):
            duplicate = next(n for n in transitions_list if transitions_list.count(n) > 1)
            return f"state {state['name']!r}: transition name {duplicate!r} is used more than once"

        on_arrival = state.get("on_arrival", None)
        if on_arrival is not None and (not isinstance(on_arrival, str) or on_arrival not in transitions_set):
            return (f"state {state['name']!r}: on_arrival {on_arrival!r} is not one of its transitions "
                    f"{sorted(transitions_set)}")

    # Rule 6
    initial_state = doc.get("initial_state", None)
    if initial_state not in state_names_set:
        if initial_state is None:
            return "initial_state is missing"
        return f"initial_state {initial_state!r} is not a state"

    return None


def arrived(state: Dict[str, Any], positions: Dict[str, float]) -> bool:
    """True when every joint named in `state["target"]` is within
    `state["epsilon"]` of its target position in `positions` (the latest
    `/joint_states`, as name -> position). `state` has already passed
    `validate_fsm`."""
    target = state['target']
    for joint_name, target_pos in zip(target["joint_names"], target["positions"]):
        if joint_name not in positions:
            return False
        actual_pos = positions[joint_name]
        if abs(actual_pos - target_pos) > state['epsilon']:
            return False
    return True

