"""Gymnasium's env checker plus the contracts of the Isaac stack (the VLNverse
lines). Boots Isaac Sim: needs the data roots, the Isaac launcher
(``EMBODIEDSCORE_ISAAC_PYTHON`` or ``~/isaacsim5.1/python.sh``), a GPU, and
``EMBODIEDSCORE_RUN_ISAAC_TESTS=1`` to opt in (a scene takes up to 3 min to load)."""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("EMBODIEDSCORE_RUN_ISAAC_TESTS") != "1" or not (
        os.environ.get("EMBODIEDSCORE_DATA_ROOT") and os.environ.get("EMBODIEDSCORE_SCENE_ROOT")):
    pytest.skip("set EMBODIEDSCORE_RUN_ISAAC_TESTS=1 and the EMBODIEDSCORE_* roots to run the Isaac contracts",
                allow_module_level=True)

from gymnasium.utils.env_checker import check_env

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks.env import VLNVERSE_KEYS, Act

NAME, SPLIT = "vlnverse-fine", "val_unseen"


@pytest.fixture(scope="module")
def env():
    e = es.make(NAME, SPLIT)
    yield e
    e.close()


def test_reset_facts_and_spaces(env):
    b = es.benchmark(NAME)
    obs, info = env.reset(options={"episode": 0})
    assert env.observation_space.contains(obs)
    assert obs["rgb"].dtype == np.uint8 and obs["rgb"].shape == (b.body.rgb.height, b.body.rgb.width, 3)
    # the STANDARD body: depth 256² next to RGB 512², post-processed to [0, 1] like every standard line
    assert obs["depth"].dtype == np.float32 and obs["depth"].shape == (b.body.depth.height, b.body.depth.width, 1)
    assert np.isfinite(obs["depth"]).all() and obs["depth"].min() >= 0.0 and obs["depth"].max() <= 1.0
    for k in ("position", "rotation", "heading", "pitch", "collided", "distance_to_goal", "stop_called", "goal_index",
              "episode", "goal", "metrics"):
        assert k in info, k
    assert info["episode"]["index"] == 0 and info["stop_called"] is False and info["goal_index"] == 0
    assert info["position"][2] == pytest.approx(b.body.camera_height_m)
    assert set(VLNVERSE_KEYS) <= set(info["metrics"])
    assert env.action_space.n == len(b.actions)


def test_step_facts_and_termination(env):
    env.reset(options={"episode": 0})
    for a in (Act.FORWARD, Act.LEFT, Act.RIGHT):
        _, r, term, trunc, info = env.step(a)
        assert r == 0.0 and term is False and trunc is False
    assert info["metrics"]["steps_taken"] == 3
    assert info["metrics"]["path_length"] > 0
    _, _, term, _, info = env.step(Act.STOP)
    assert term is True and info["stop_called"] is True
    with pytest.raises(RuntimeError):
        env.step(Act.FORWARD)


def test_panorama_does_not_move_the_agent(env):
    env.reset(options={"episode": 0})
    world = env.unwrapped.world
    before = world.pose()[0].copy()
    views = world.render_views(4)
    assert len(views) == 4 and all(v["rgb"].shape == views[0]["rgb"].shape for v in views)
    assert np.allclose(world.pose()[0], before)


def test_episode_selection_and_seeding(env):
    _, info = env.reset(options={"episode": 2})
    assert info["episode"]["index"] == 2
    a = env.reset(seed=123)[1]["episode"]["index"]
    b = env.reset(seed=123)[1]["episode"]["index"]
    assert a == b


def test_same_episode_twice_facts_exact_pixels_close(env):
    """What ``nondeterministic=True`` in the registration means: the facts and the
    depth frame are bit-identical across resets of one episode; the RGB frame is
    not (RTX real-time lighting). Measured 2026-09-07 on kujiale_0272, episode 0:
    rgb mean |diff| 3.5–5.8 grey levels, max ~99, 87–90 % of pixels differ by >= 1;
    depth max |diff| 0.0; consecutive renders without a reset: mean 1.5."""
    obs1, info1 = env.reset(options={"episode": 0})
    obs2, info2 = env.reset(options={"episode": 0})
    for k in ("position", "rotation", "heading", "distance_to_goal", "episode"):
        assert info1[k] == info2[k], k
    assert np.array_equal(obs1["depth"], obs2["depth"])
    d = np.abs(obs1["rgb"].astype(np.int16) - obs2["rgb"].astype(np.int16)).mean()
    assert d < 12.0, f"rgb mean |diff| between two renders of one pose = {d:.2f} grey levels"


def test_gym_checker(env):
    env.reset(options={"episode": 0})
    check_env(env.unwrapped, skip_render_check=True)


def test_upstream_is_the_zero_shot_rig_behind_the_polar_action(env):
    """``-upstream``: RGB-D 1024² at 1.2 m, depth in metres, Box[angle, distance,
    elevation]; (0, 0) is STOP. Closes the module's stack first — one Isaac at a
    time — so this is the last test of the module."""
    env.close()
    b = es.benchmark(NAME, "upstream")
    up = es.make(NAME, SPLIT, variant="upstream")
    try:
        obs, info = up.reset(options={"episode": 0})
        assert obs["rgb"].shape == (1024, 1024, 3) and obs["depth"].shape == (1024, 1024, 1)
        assert obs["depth"].max() > 1.0                      # metres, not normalised
        assert info["position"][2] == pytest.approx(b.body.camera_height_m) == 1.2
        assert up.action_space.shape == (3,)
        _, r, term, trunc, info = up.step([0.0, 0.25, 0.0])
        assert r == 0.0 and term is False and "deviation_m" in info and info["metrics"]["steps_taken"] == 1
        _, _, term, _, info = up.step([0.0, 0.0, 0.0])
        assert term is True and info["stop_called"] is True
    finally:
        up.close()
