"""``VLNVerseMetrics`` against the evaluator's own ``VLNPEMetrics`` (the verbatim
oracle in ``oracle_vlnpe_metrics.py``), driven with the same trajectories; plus
the quirks the formula set carries, pinned so a "fix" is a deliberate act."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import Env, spaces

from embodiedscore_envs.benchmarks.env.metrics import VLNVERSE_EVALUATOR_NAMES as A_NAMES, VLNVERSE_KEYS, VLNVerseMetrics

from .oracle_vlnpe_metrics import Cfg, VLNPEMetrics

Z = 1.2


class ScriptedEnv(Env):
    """A body stand-in: replays a list of positions; action 0 is STOP."""

    def __init__(self, episode: dict, positions):
        self.episode_dict = episode
        self.positions = [np.asarray(p, dtype=np.float64) for p in positions]
        self.observation_space = spaces.Dict({"rgb": spaces.Box(0, 255, shape=(2, 2, 3), dtype=np.uint8)})
        self.action_space = spaces.Discrete(4)
        self._i = 0
        self._stop = False

    def _facts(self, pos):
        return {"position": list(pos), "rotation": [1.0, 0.0, 0.0, 0.0], "heading": 0.0, "pitch": 0.0,
                "collided": False, "distance_to_goal": None, "stop_called": self._stop}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._i, self._stop = 0, False
        start = list(self.episode_dict["start_position"])
        start[2] = Z
        info = self._facts(start)
        info["episode"] = self.episode_dict
        return {"rgb": np.zeros((2, 2, 3), np.uint8)}, info

    def step(self, action):
        if int(action) == 0:
            self._stop = True
            pos = self.positions[self._i - 1] if self._i else np.array(self.episode_dict["start_position"]) * [1, 1, 0] + [0, 0, Z]
        else:
            pos = self.positions[self._i]
            self._i += 1
        return {"rgb": np.zeros((2, 2, 3), np.uint8)}, 0.0, self._stop, False, self._facts(pos)


def _episode(ref, eid="ep0"):
    return {"episode_id": eid, "trajectory_id": eid, "start_position": list(ref[0]), "reference_path": [list(p) for p in ref],
            "goals": {"position": list(ref[-1]), "radius": 3.0}, "info": {"geodesic_distance": -1}, "instruction": "go"}


def _oracle(episode, positions):
    m = VLNPEMetrics(Cfg(success_distance=3.0, shortest_to_goal_distance=999), Cfg(data=episode))
    start = np.array(episode["start_position"], dtype=np.float64)
    m.update({"h1": {"globalgps": start, "finish_action": False}})      # the warm-up update seats the start
    for p in positions:
        m.update({"h1": {"globalgps": np.asarray(p, dtype=np.float64), "finish_action": True}})
    return m.calc()[0]


def _ours(episode, positions, stop=True):
    env = VLNVerseMetrics(ScriptedEnv(episode, positions))
    _, info = env.reset()
    for _ in positions:
        _, _, _, _, info = env.step(1)
    if stop:
        _, _, _, _, info = env.step(0)
    return info


def _random_case(rng):
    n_ref = int(rng.integers(3, 9))
    ref = np.cumsum(rng.normal(0, 1.0, size=(n_ref, 3)) * [1, 1, 0], axis=0)
    n = int(rng.integers(1, 15))
    positions = np.cumsum(rng.normal(0, 0.6, size=(n, 3)) * [1, 1, 0], axis=0) + ref[0]
    positions[:, 2] = Z
    return _episode(ref.tolist()), positions.tolist()


@pytest.mark.parametrize("seed", range(25))
def test_parity_with_the_evaluator(seed):
    episode, positions = _random_case(np.random.default_rng(seed))
    theirs = _oracle(episode, positions)
    ours = _ours(episode, positions, stop=False)["metrics"]
    for key in VLNVERSE_KEYS:
        if key == "steps_taken":
            continue        # the evaluator counts simulator updates (warm-up included); documented
        assert ours[key] == pytest.approx(theirs[A_NAMES[key]], abs=1e-9), key


def test_stop_does_not_change_the_numbers():
    """STOP is a step at the same position: NE / success / TL unchanged, ndtw
    sees one more (identical) trajectory point — as the evaluator's
    finish_action bookkeeping would."""
    episode, positions = _random_case(np.random.default_rng(99))
    without = _ours(episode, positions, stop=False)["metrics"]
    with_stop = _ours(episode, positions, stop=True)["metrics"]
    for key in ("distance_to_goal", "success", "oracle_success", "spl", "path_length", "shortest_path_length"):
        assert with_stop[key] == pytest.approx(without[key])
    assert with_stop["steps_taken"] == without["steps_taken"] + 1
    theirs = _oracle(episode, positions + [positions[-1]])
    assert with_stop["ndtw"] == pytest.approx(theirs["ndtw"])


def test_success_without_stop_and_strict_radius():
    ref = [[0, 0, 0], [5, 0, 0]]
    ep = _episode(ref)
    m = _ours(ep, [[2.0, 0, Z]], stop=False)["metrics"]        # 3.0 m away: NOT a success (strict <)
    assert m["distance_to_goal"] == pytest.approx(3.0) and m["success"] == 0.0
    m = _ours(ep, [[2.5, 0, Z]], stop=False)["metrics"]        # 2.5 m away, no STOP: a success
    assert m["success"] == 1.0


def test_spl_zero_without_movement_and_start_excluded_from_oracle():
    ref = [[0, 0, 0], [2, 0, 0]]                                # start already within 3 m of the goal
    ep = _episode(ref)
    env = VLNVerseMetrics(ScriptedEnv(ep, [[0.0, 10.0, Z]]))
    _, info = env.reset()
    assert info["metrics"]["success"] == 1.0                    # this wrapper's addition: NE at reset
    assert info["metrics"]["oracle_success"] == 0.0             # the evaluator ignores the start pose
    _, _, _, _, info = env.step(0)                              # STOP without moving
    assert info["metrics"]["success"] == 1.0 and info["metrics"]["spl"] == 0.0   # TL == 0 -> spl 0
    _, info = env.reset()
    _, _, _, _, info = env.step(1)                              # move 10 m away
    assert info["metrics"]["oracle_success"] == 0.0             # the start's 2 m never counted


def test_ndtw_is_the_nearest_point_similarity_not_a_dtw():
    ref = [[0, 0, 0], [10, 0, 0]]
    ep = _episode(ref)
    m = _ours(ep, [[10.0, 0.0, Z]], stop=False)["metrics"]      # jump straight to the end
    assert m["ndtw"] == pytest.approx(1.0)                      # both points sit on the reference: 1.0
    m = _ours(ep, [[3.0, 3.0, Z]], stop=False)["metrics"]       # nearest WAYPOINT is [0, 0]: d^2 = 18, not the
    assert m["ndtw"] == pytest.approx((1.0 + np.exp(-18.0 / 18.0)) / 2)   # 3 m to the segment


def test_shortest_path_is_the_reference_polyline_when_geodesic_missing():
    ref = [[0, 0, 0], [3, 4, 0], [3, 10, 0]]
    m = _ours(_episode(ref), [[3.0, 10.0, Z]], stop=False)["metrics"]
    assert m["shortest_path_length"] == pytest.approx(11.0)
    assert m["spl"] == pytest.approx(11.0 / max(np.hypot(3, 10), 11.0))
    ep = _episode(ref)
    ep["info"]["geodesic_distance"] = 12.5
    assert _ours(ep, [[3.0, 10.0, Z]], stop=False)["metrics"]["shortest_path_length"] == pytest.approx(12.5)


def test_test_split_placeholders():
    ep = {"episode_id": "t0", "trajectory_id": "t0", "start_position": [0, 0, 0], "instruction": "go"}
    info = _ours(ep, [[1.0, 0.0, Z], [2.0, 0.0, Z]], stop=True)
    m = info["metrics"]
    assert info["metrics_valid"] is False
    for key in ("distance_to_goal", "success", "oracle_success", "spl", "ndtw", "shortest_path_length"):
        assert m[key] == -1.0
    assert m["path_length"] == pytest.approx(2.0) and m["steps_taken"] == 3
    theirs = _oracle(ep, [[1.0, 0.0, Z], [2.0, 0.0, Z]])
    assert theirs["NE"] == -1 and theirs["TL"] == pytest.approx(2.0)


def test_key_selection():
    ep = _episode([[0, 0, 0], [1, 0, 0]])
    env = VLNVerseMetrics(ScriptedEnv(ep, [[1.0, 0.0, Z]]), keys=("success", "path_length"))
    _, info = env.reset()
    assert set(info["metrics"]) == {"success", "path_length"}
    with pytest.raises(ValueError):
        VLNVerseMetrics(ScriptedEnv(ep, []), keys=("colrate",))
