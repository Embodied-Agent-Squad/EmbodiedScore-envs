"""Gymnasium's env checker plus the space / semantics contract of VLNCEEnv.
Needs habitat-sim, the datasets and a GPU: skipped unless the data roots are set."""

import os

import numpy as np
import pytest

pytest.importorskip("habitat_sim")
if not (os.environ.get("EMBODIEDSCORE_DATA_ROOT") and os.environ.get("EMBODIEDSCORE_SCENE_ROOT")):
    pytest.skip("EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT not set", allow_module_level=True)

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import embodiedscore_envs  # noqa: F401
from embodiedscore_envs.vlnce import Act, VLNCEEnv, make_vlnce

SPLIT = os.environ.get("EMBODIEDSCORE_TEST_SPLIT", "val_unseen")


@pytest.fixture(scope="module")
def body_env():
    env = VLNCEEnv(dataset="r2r", split=SPLIT)
    yield env
    env.close()


def test_check_env(body_env):
    check_env(body_env, skip_render_check=True)


def test_reset_facts_and_spaces(body_env):
    obs, info = body_env.reset(seed=0, options={"episode": 0})
    assert body_env.observation_space.contains(obs)
    assert obs["rgb"].dtype == np.uint8 and obs["depth"].dtype == np.float32
    for k in ("position", "rotation", "heading", "collided", "distance_to_goal", "stop_called", "episode"):
        assert k in info
    assert info["stop_called"] is False and info["collided"] is False
    assert info["episode"]["index"] == 0


def test_stop_terminates_and_body_has_no_budget(body_env):
    body_env.reset(options={"episode": 0})
    _, r, term, trunc, info = body_env.step(Act.STOP)
    assert r == 0.0 and term is True and trunc is False and info["stop_called"] is True
    with pytest.raises(RuntimeError):
        body_env.step(Act.FORWARD)


def test_episode_selection(body_env):
    _, info = body_env.reset(options={"episode": 2})
    assert info["episode"]["index"] == 2
    ep_id = info["episode"]["episode_id"]
    _, info2 = body_env.reset(options={"episode_id": ep_id})
    assert info2["episode"]["episode_id"] == ep_id
    _, info3 = body_env.reset(seed=123)
    _, info4 = body_env.reset(seed=123)
    assert info3["episode"]["index"] == info4["episode"]["index"]   # seeded random choice is reproducible


def test_stack_metrics_and_depth():
    env = make_vlnce("r2r", SPLIT)
    try:
        obs, info = env.reset(options={"episode": 0})
        assert set(info["metrics"]) == {"distance_to_goal", "success", "spl", "ndtw", "path_length", "oracle_success", "steps_taken"}
        assert info["metrics"]["steps_taken"] == 0.0 and 0.0 <= obs["depth"].max() <= 1.0
        obs, _, term, trunc, info = env.step(Act.LEFT)
        assert info["metrics"]["steps_taken"] == 1.0 and info["metrics"]["path_length"] == 0.0
        assert isinstance(env, gym.Wrapper) and env.spec.max_episode_steps == 500
    finally:
        env.close()
