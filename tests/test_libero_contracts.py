"""The LIBERO lines: the declaration contracts (no simulator), and the
Gymnasium env checker plus the shared facts on one episode (needs the
``libero`` package and EGL; opt in with EMBODIEDSCORE_RUN_LIBERO_TESTS=1)."""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import libero as L
from embodiedscore_envs.benchmarks.env import LiberoEnv, LiberoPoseEnv, ManipGoal, targets_of
from embodiedscore_envs.benchmarks.presets import bodies

# The base LIBERO lines only — the LIBERO-PRO / LIBERO-Plus lines run on the same
# engine and have their own contract files (test_libero_pro / _plus_contracts.py).
LINES = sorted({f"libero-{word}" for word in L.SUITES})


def test_declarations():
    assert LINES == ["libero-10", "libero-90", "libero-goal", "libero-object", "libero-spatial"]
    for line in LINES:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert std.engine == up.engine == "libero" and std.line == up.line == line
        assert std.macro and not up.macro
        assert std.body is bodies.LIBERO_STANDARD and up.body is bodies.LIBERO
        assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
        suite = L.SUITES[line[len("libero-"):]]
        assert up.max_episode_steps == up.ticks == L.TICKS[suite]
        assert std.max_episode_steps == L.MACRO_STEPS and std.ticks == L.TICKS[suite] * L.TICK_GUARD
        assert std.splits == up.splits == L.SPLITS
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")


def test_manip_goal_has_no_place():
    g = ManipGoal((("on", "bowl", "plate"),))
    assert g.kind == "manip" and targets_of(g) is None


def test_hold_action_is_a_hold():
    a = LiberoPoseEnv.hold_action(+1)
    assert a.shape == (7,) and not np.isfinite(a[:3]).any() and a[6] == 1.0


@pytest.fixture(scope="module")
def stack():
    pytest.importorskip("libero")
    if not os.environ.get("EMBODIEDSCORE_RUN_LIBERO_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_LIBERO_TESTS not set")
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = es.make("libero-spatial", "mini")
    yield env
    env.close()


def test_mini_split_and_episode_facts(stack):
    base = stack.unwrapped
    assert isinstance(base, LiberoPoseEnv) and isinstance(base, LiberoEnv)
    assert len(base.episodes) == 10 * L.MINI_INIT_STATES
    ep = base.episodes[0]
    assert ep.episode_id == "libero_spatial/0/0" and ep.start_position is None and ep.goal.kind == "manip"
    assert ep.instruction and ep.info["init_state_index"] == 0
    assert base.episodes[L.MINI_INIT_STATES].info["task_id"] == 1


def test_reset_step_hold_facts(stack):
    obs, info = stack.reset(options={"episode": 0})
    assert set(obs) == {"rgb", "wrist", "proprio"} and obs["rgb"].shape == (256, 256, 3) and obs["proprio"].shape == (8,)
    assert info["metrics"] == {"success": 0.0, "steps_taken": 0, "ticks": bodies.LIBERO_STANDARD.settle_ticks}
    assert info["success"] is False and len(info["eef_position"]) == 3 and len(info["eef_rotation"]) == 4
    assert info["gripper_open"] > 90 and info["objects"] and len(info["robot_base"]) == 3
    pos = np.asarray(info["eef_position"])
    from scipy.spatial.transform import Rotation
    w, x, y, z = info["eef_rotation"]
    rot = Rotation.from_quat([x, y, z, w]).as_rotvec()
    obs, r, term, trunc, info = stack.step(np.concatenate([pos + [0.0, 0.0, -0.05], rot, [-1.0]]))
    assert r == 0.0 and not term and not trunc and info["converged"] and info["position_error_m"] < 0.016
    assert info["metrics"]["steps_taken"] == 1 and info["metrics"]["success"] == 0.0
    obs, r, term, trunc, info = stack.step(LiberoPoseEnv.hold_action(+1))
    assert info["gripper_open"] < 10 and info["metrics"]["steps_taken"] == 2


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env
    check_env(stack.unwrapped, skip_render_check=True)


def test_upstream_protocol_truncates_at_the_tick_cap():
    pytest.importorskip("libero")
    if not os.environ.get("EMBODIEDSCORE_RUN_LIBERO_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_LIBERO_TESTS not set")
    env = es.make("libero-spatial-upstream", "mini")
    try:
        obs, info = env.reset(options={"episode": 0})
        assert obs["rgb"].shape == (128, 128, 3) and isinstance(env.unwrapped, LiberoEnv)
        n, trunc = 0, False
        while not trunc:
            obs, r, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
            n += 1
            assert not term
        assert n == L.TICKS["libero_spatial"] and info["metrics"]["steps_taken"] == n
    finally:
        env.close()
