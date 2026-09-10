"""The RoboTwin lines: the declaration contracts (no simulator), and the
Gymnasium env checker plus the shared facts on one episode (needs the
RoboTwin checkout, its dependencies and a Vulkan GPU; opt in with
EMBODIEDSCORE_RUN_ROBOTWIN_TESTS=1)."""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import robotwin as R
from embodiedscore_envs.benchmarks.env import ManipGoal, RobotwinEnv, RobotwinPoseEnv, targets_of
from embodiedscore_envs.benchmarks.env.robotwin_env import ARM_ACTION, gripper_to_robotwin
from embodiedscore_envs.benchmarks.presets import bodies

LINES = sorted({b.line for b in es.BENCHMARKS.values() if b.engine == "robotwin"})


def test_declarations():
    assert LINES == ["robotwin-clean", "robotwin-randomized"]
    for line in LINES:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert std.engine == up.engine == "robotwin" and std.line == up.line == line
        assert std.macro and not up.macro
        assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
        assert std.ticks is None and up.ticks is None          # the cap is per task, through ``budget``
        assert std.budget is R.step_budget and up.budget is R.step_budget
        assert std.max_episode_steps == R.MACRO_STEPS and up.max_episode_steps is None
        assert up.truncate_at_budget and not std.truncate_at_budget
        assert std.splits == up.splits == R.SPLITS
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")
    assert es.benchmark("robotwin-clean").body is bodies.ROBOTWIN_STANDARD
    assert es.benchmark("robotwin-clean", "upstream").body is bodies.ROBOTWIN
    assert es.benchmark("robotwin-randomized").body is bodies.ROBOTWIN_STANDARD_RANDOMIZED
    assert es.benchmark("robotwin-randomized", "upstream").body is bodies.ROBOTWIN_RANDOMIZED


def test_the_fifty_tasks_and_their_caps():
    assert len(R.TASKS) == 50 and len(R.STEP_LIMIT) == 50
    assert set(R.TASKS) == set(R.STEP_LIMIT)
    assert min(R.STEP_LIMIT.values()) == 400 and max(R.STEP_LIMIT.values()) == 1700
    assert R.STEP_LIMIT["beat_block_hammer"] == 400 and R.STEP_LIMIT["put_bottles_dustbin"] == 1700


def test_randomization_presets_are_the_two_task_configs():
    assert bodies.CLEAN.random_background is False and bodies.CLEAN.clean_background_rate == 1.0
    assert bodies.CLEAN.cluttered_table is False and bodies.CLEAN.random_light is False
    assert bodies.RANDOMIZED.random_background and bodies.RANDOMIZED.cluttered_table
    assert bodies.RANDOMIZED.clean_background_rate == 0.02 and bodies.RANDOMIZED.random_table_height == 0.03
    assert bodies.RANDOMIZED.random_light and bodies.RANDOMIZED.crazy_random_light_rate == 0.02
    assert bodies.ROBOTWIN.randomization is bodies.CLEAN
    assert bodies.ROBOTWIN_RANDOMIZED.randomization is bodies.RANDOMIZED


def test_body_action_widths():
    assert bodies.ROBOTWIN.qpos_dim == 14 and bodies.ROBOTWIN.ee_dim == 16      # aloha-agilex: 6 + 1 per arm
    assert bodies.ROBOTWIN.head.width == 320 and bodies.ROBOTWIN.head.height == 240
    assert bodies.ROBOTWIN_STANDARD.head.width == 640 and bodies.ROBOTWIN_STANDARD.head.height == 480
    assert bodies.ROBOTWIN.depth is None and bodies.ROBOTWIN.rgb is bodies.ROBOTWIN.head


def test_manip_goal_has_no_place():
    g = ManipGoal((("check_success", "beat_block_hammer"),))
    assert g.kind == "manip" and targets_of(g) is None


def test_hold_action_holds_both_arms():
    a = RobotwinPoseEnv.hold_action(+1, -1)
    assert a.shape == (2 * ARM_ACTION,)
    assert not np.isfinite(a[:3]).any() and not np.isfinite(a[ARM_ACTION:ARM_ACTION + 3]).any()
    assert a[6] == 1.0 and a[13] == -1.0


def test_arm_action_moves_one_arm_and_holds_the_other():
    left = RobotwinPoseEnv.arm_action("left", [0.1, 0.2, 0.9], [0.0, 0.0, 0.0], gripper=-1)
    assert np.isfinite(left[:3]).all() and not np.isfinite(left[ARM_ACTION:ARM_ACTION + 3]).any()
    right = RobotwinPoseEnv.arm_action("right", [0.1, 0.2, 0.9], [0.0, 0.0, 0.0], gripper=+1)
    assert not np.isfinite(right[:3]).any() and np.isfinite(right[ARM_ACTION:ARM_ACTION + 3]).all()
    with pytest.raises(ValueError):
        RobotwinPoseEnv.arm_action("middle", [0, 0, 0], [0, 0, 0], gripper=0)


def test_gripper_convention_matches_liberos_signs():
    assert gripper_to_robotwin(+1) == 0.0        # closed
    assert gripper_to_robotwin(-1) == 1.0        # open
    assert gripper_to_robotwin(0.0) == 0.5       # half, which RoboTwin's [0, 1] range allows


def test_episode_seeds_are_robotwins_own(monkeypatch):
    monkeypatch.setattr(R, "task_instruction", lambda scene, root=None: scene.task.replace("_", " "))
    eps = R.load_episodes("demo_clean", "mini")
    assert len(eps) == 50 * R.MINI_SEEDS
    assert eps[0].info["task"] == R.TASKS[0] and eps[0].info["seed"] == R.SEED_BASE
    assert eps[0].episode_id == f"{R.TASKS[0]}/{R.SEED_BASE}"
    assert eps[0].start_position is None and eps[0].start_rotation is None
    assert eps[0].goal.kind == "manip" and eps[0].info["step_limit"] == R.STEP_LIMIT[eps[0].info["task"]]
    # task-major, and each episode owns its own seed window
    assert eps[1].info["seed"] == R.SEED_BASE + R.SEED_STRIDE and eps[1].info["seed_window"] == R.SEED_STRIDE
    assert eps[R.MINI_SEEDS].info["task"] != eps[0].info["task"]
    assert eps[R.MINI_SEEDS].info["seed"] == R.SEED_BASE      # every task restarts from the base
    seeds = {(e.info["task"], s) for e in eps for s in range(e.info["seed"], e.info["seed"] + R.SEED_STRIDE)}
    assert len(seeds) == len(eps) * R.SEED_STRIDE            # the windows never overlap
    assert R.step_budget(None, eps[0]) == eps[0].info["step_limit"]
    with pytest.raises(ValueError):
        R.load_episodes("demo_clean", "nope")
    with pytest.raises(ValueError):
        R.load_episodes("demo_nope", "mini")


# ---- the simulator half ---------------------------------------------------------------------

def _opted_in() -> bool:
    return bool(os.environ.get("EMBODIEDSCORE_RUN_ROBOTWIN_TESTS"))


@pytest.fixture(scope="module")
def stack():
    if not _opted_in():
        pytest.skip("EMBODIEDSCORE_RUN_ROBOTWIN_TESTS not set")
    pytest.importorskip("sapien")
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    env = es.make("robotwin-clean", "mini")
    yield env
    env.close()


def test_mini_split_and_episode_facts(stack):
    base = stack.unwrapped
    assert isinstance(base, RobotwinPoseEnv) and isinstance(base, RobotwinEnv)
    assert len(base.episodes) == 50 * R.MINI_SEEDS
    ep = base.episodes[0]
    assert ep.start_position is None and ep.goal.kind == "manip" and ep.instruction
    assert ep.info["seed"] == R.SEED_BASE and ep.info["embodiment"] == R.EMBODIMENT


def test_reset_step_hold_facts(stack):
    body = bodies.ROBOTWIN_STANDARD
    obs, info = stack.reset(options={"episode": 0})
    assert obs["rgb"].shape == (body.head.height, body.head.width, 3)
    assert obs["proprio"].shape == (body.qpos_dim,)
    assert set(obs) >= {"rgb", "left_wrist", "right_wrist", "proprio"}
    assert info["metrics"] == {"success": 0.0, "steps_taken": 0, "ticks": 0}
    assert info["success"] is False and set(info["arms"]) == {"left", "right"}
    assert info["step_budget"] == R.STEP_LIMIT[stack.unwrapped.episode.info["task"]]
    for arm in ("left", "right"):
        assert len(info["arms"][arm]["eef_position"]) == 3 and len(info["arms"][arm]["eef_rotation"]) == 4
        assert info["arms"][arm]["gripper_open"] > 90        # RoboTwin opens both grippers at setup
    assert info["objects"] and set(info["robot_base"]) == {"left", "right"}

    # a hold closes the left gripper and moves neither arm
    before = np.asarray(info["arms"]["left"]["eef_position"])
    obs, r, term, trunc, info = stack.step(RobotwinPoseEnv.hold_action(left_gripper=+1, right_gripper=-1))
    assert r == 0.0 and not term and info["converged"]
    assert info["arms"]["left"]["gripper_open"] < 50 and info["arms"]["right"]["gripper_open"] > 50
    assert np.linalg.norm(np.asarray(info["arms"]["left"]["eef_position"]) - before) < 0.05
    assert info["metrics"]["steps_taken"] == 1


def test_a_macro_move_reaches_its_target(stack):
    obs, info = stack.reset(options={"episode": 0})
    pos = np.asarray(info["arms"]["left"]["eef_position"], dtype=np.float64)
    from scipy.spatial.transform import Rotation
    w, x, y, z = info["arms"]["left"]["eef_rotation"]
    rotvec = Rotation.from_quat([x, y, z, w]).as_rotvec()
    obs, r, term, trunc, info = stack.step(
        RobotwinPoseEnv.arm_action("left", pos + [0.0, 0.0, 0.05], rotvec, gripper=-1))
    assert info["moves"]["left"]["commanded"] and not info["moves"]["right"]["commanded"]
    assert info["converged"] and info["position_error_m"] < 0.02
    assert info["ticks"] >= 1


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env

    check_env(stack.unwrapped, skip_render_check=True)


def test_upstream_protocol_holds_its_pose_under_a_repeated_joint_target():
    if not _opted_in():
        pytest.skip("EMBODIEDSCORE_RUN_ROBOTWIN_TESTS not set")
    pytest.importorskip("sapien")
    os.environ.setdefault("SAPIEN_HEADLESS", "1")
    env = es.make("robotwin-clean-upstream", "mini")
    try:
        obs, info = env.reset(options={"episode": 0})
        base = env.unwrapped
        assert isinstance(base, RobotwinEnv) and not isinstance(base, RobotwinPoseEnv)
        assert base.action_type == "qpos"
        assert obs["rgb"].shape == (bodies.ROBOTWIN.head.height, bodies.ROBOTWIN.head.width, 3)
        action = np.asarray(obs["proprio"], dtype=np.float64)     # hold the joint state it is already in
        assert action.shape == (bodies.ROBOTWIN.qpos_dim,)
        for n in range(3):
            obs, r, term, trunc, info = env.step(action)
            assert r == 0.0 and info["metrics"]["steps_taken"] == n + 1 and info["ticks"] == n + 1
    finally:
        env.close()
