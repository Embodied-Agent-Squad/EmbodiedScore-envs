"""Gymnasium's env checker plus the contracts every benchmark shares — the
habitat lines here, the Isaac lines in ``test_vlnverse_contracts.py`` (they boot
Isaac Sim and opt in separately), the LIBERO lines in ``test_libero_contracts.py``, the RoboCasa lines in
``test_robocasa_contracts.py``. Needs habitat-sim, the datasets and a GPU:
skipped unless the data roots are set."""

import os

import numpy as np
import pytest

pytest.importorskip("habitat_sim")
if not (os.environ.get("EMBODIEDSCORE_DATA_ROOT") and os.environ.get("EMBODIEDSCORE_SCENE_ROOT")):
    pytest.skip("EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT not set", allow_module_level=True)

from gymnasium.utils.env_checker import check_env

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks.env import Act

# one cheap split per line (both variants share it)
SPLITS = {
    "vlnce-r2r": "val_unseen", "vlnce-rxr": "val_unseen", "ivlnce": "val_unseen",
    "objectnav-hm3d-v1": "val_mini", "objectnav-hm3d-v2": "val_mini", "objectnav-mp3d-v1": "val_mini",
    "ovon": "mip100_seen", "goat": "val_unseen",
    "hmeqa": "mip100", "mthm3d": "mip100", "express": "mip100",
}


def split_of(name: str) -> str:
    return SPLITS[es.benchmark(name).line]


HABITAT = sorted(n for n, b in es.BENCHMARKS.items() if b.engine == "habitat")


@pytest.fixture(scope="module", params=HABITAT)
def stack(request):
    name = request.param
    env = es.make(name, split_of(name))
    yield name, env
    env.close()


def test_registry_covers_every_declaration():
    import gymnasium as gym
    for b in es.BENCHMARKS.values():
        assert b.gym_id in gym.registry


def test_every_line_has_both_variants():
    from embodiedscore_envs.benchmarks import resolve
    from embodiedscore_envs.benchmarks.presets import actions, bodies, depth
    lines = {b.line for b in es.BENCHMARKS.values()}
    assert len(es.BENCHMARKS) == 2 * len(lines) == 84
    assert len(HABITAT) == 22
    for line in lines:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert (std.variant, up.variant) == ("standard", "upstream") and std.line == up.line == line
        assert std.engine == up.engine
        if std.engine == "habitat":
            assert std.body is bodies.STANDARD and std.actions in (actions.STANDARD, actions.GOAT)
            assert std.depth is depth.STANDARD
        elif std.engine == "isaac":   # the Isaac worker renders yaw-only poses: STANDARD's numbers, no LOOK
            assert std.body is bodies.VLNVERSE_STANDARD and std.actions == actions.NAV
            assert std.depth is depth.STANDARD
        elif std.engine == "libero":   # no depth camera, no action table (test_libero_contracts.py has the rest)
            assert std.body is bodies.LIBERO_STANDARD and std.depth is None
        elif std.engine == "robotwin":   # likewise (test_robotwin_contracts.py has the rest)
            assert std.depth is None and std.macro
            assert std.body in (bodies.ROBOTWIN_STANDARD, bodies.ROBOTWIN_STANDARD_RANDOMIZED)
        else:   # robocasa: likewise (test_robocasa_contracts.py has the rest)
            assert std.engine == "robocasa" and std.body is bodies.ROBOCASA_STANDARD and std.depth is None
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")
        assert es.benchmark(f"{line}-upstream", "standard") is std
    assert resolve("goat-upstream") == "goat-upstream" and resolve("goat-upstream", "standard") == "goat"
    with pytest.raises(ValueError):
        resolve("goat", "legacy")
    with pytest.raises(KeyError):
        es.benchmark("goat-legacy")


def test_ovon_upstream_protocol_widens_goal_with_children():
    """OVONDistanceToGoal's target set = the category's view points + those of
    every children category present in the shard; both variants carry it."""
    from embodiedscore_envs.benchmarks.env import targets_of
    for name in ("ovon", "ovon-upstream"):
        b = es.benchmark(name)
        eps = b.episodes(SPLITS["ovon"])
        widened = [e for e in eps if e.info.get("children_object_categories")]
        assert widened, "mip100_seen has no episode with children categories"
        e = widened[0]
        own = {i.object_id for i in e.goal.instances if i.category == e.goal.category}
        assert own and len(e.goal.instances) >= len(own)
        assert targets_of(e.goal).shape[1] == 3


def test_check_env_and_spaces(stack):
    name, env = stack
    b = es.benchmark(name)
    obs, info = env.reset(options={"episode": 0})
    assert env.observation_space.contains(obs), name
    assert obs["rgb"].dtype == np.uint8 and obs["rgb"].shape[:2] == (b.body.rgb.height, b.body.rgb.width)
    if b.body.depth is not None:
        assert obs["depth"].dtype == np.float32 and obs["depth"].shape[-1] == 1
    for k in ("position", "rotation", "heading", "pitch", "collided", "distance_to_goal", "stop_called", "goal_index", "episode", "goal"):
        assert k in info, (name, k)
    assert info["episode"]["index"] == 0 and info["stop_called"] is False
    if b.metrics is not None:
        assert "metrics" in info
    if not b.pose:
        assert env.action_space.n == len(b.actions)
        check_env(env.unwrapped, skip_render_check=True)


def test_step_facts_and_termination(stack):
    name, env = stack
    b = es.benchmark(name)
    env.reset(options={"episode": 1})
    if b.pose:
        _, info = env.reset(options={"episode": 1})
        x, _, z = info["position"]
        _, r, term, trunc, info2 = env.step([x, z, info["heading"]])
        assert r == 0.0 and term is False and "step_geodesic" in info2
        return
    for a in (Act.FORWARD, Act.LEFT, Act.RIGHT):
        _, r, term, trunc, info = env.step(a)
        assert r == 0.0 and term is False and trunc is False
    if Act.LOOK_UP in b.actions:
        p0 = info["pitch"]
        _, _, _, _, info = env.step(Act.LOOK_UP)
        assert info["pitch"] > p0 - 1e-6
    _, _, term, trunc, info = env.step(Act.STOP)
    assert term is True and info["stop_called"] is True
    with pytest.raises(RuntimeError):
        env.step(Act.FORWARD)


def test_episode_selection_and_seeding(stack):
    name, env = stack
    _, info = env.reset(options={"episode": 2})
    assert info["episode"]["index"] == 2
    if es.benchmark(name).line != "ivlnce":   # tours address episodes explicitly
        a = env.reset(seed=123)[1]["episode"]["index"]
        b = env.reset(seed=123)[1]["episode"]["index"]
        assert a == b


@pytest.mark.parametrize("variant", ["standard", "upstream"])
def test_goal_sequence_advances(variant):
    env = es.make("goat", SPLITS["goat"], variant=variant)
    try:
        _, info = env.reset(options={"episode": 0})
        n = info["metrics"]["n_subtasks"]
        assert info["goal_index"] == 0 and n >= 5
        _, _, term, _, info = env.step(Act.SUBTASK_STOP)
        assert info["goal_index"] == 1 and info["subtask_closed"] is True and term is False
        assert len(info["metrics"]["subtask_success"]) == n
    finally:
        env.close()


def test_dynamic_budget_truncates_pose_env():
    env = es.make("hmeqa", SPLITS["hmeqa"], variant="upstream")
    try:
        _, info = env.reset(options={"episode": 0})
        budget = info["step_budget"]
        assert budget > 0
        x, _, z = info["position"]
        trunc = False
        for k in range(budget):
            _, _, _, trunc, _ = env.step([x, z, 0.0])
            if k < budget - 1:
                assert trunc is False
        assert trunc is True
    finally:
        env.close()


def test_tour_transit():
    env = es.make("ivlnce", SPLITS["ivlnce"])
    try:
        _, info = env.reset(options={"episode": 0})
        assert info["tour"]["tour_index"] == 0 and info["tour"]["tour_protocol_ok"] is True
        _, _, term, _, info = env.step(Act.STOP)
        assert term and info["tour"]["tour_episodes_done"] == 1 and info["tour"]["tour_t_ndtw"] is not None
        _, info = env.reset(options={"episode": 1})
        assert info["tour"]["tour_index"] == 1 and info["tour"]["transit_steps"] > 0
        assert info["tour"]["tour_protocol_ok"] is True
        _, info = env.reset(options={"episode": 5})     # a jump breaks the protocol
        assert info["tour"]["tour_protocol_ok"] is False
    finally:
        env.close()
