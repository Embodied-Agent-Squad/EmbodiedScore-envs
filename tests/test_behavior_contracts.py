"""The BEHAVIOR-1K line: the declaration contracts (no simulator), the loader
against the release metadata and the BDDL definitions, and the Gymnasium env
checker plus the shared facts on one episode.

Three tiers, each skipping when its inputs are absent:

* declarations — always run, no data and no simulator;
* loader — needs the ``2025-challenge-task-instances`` metadata under
  ``OMNIGIBSON_DATA_PATH`` (and, for the goal predicates, the ``bddl``
  package, which is pip-installable on its own);
* simulator — needs ``omnigibson`` + Isaac Sim and a GPU; opt in with
  EMBODIEDSCORE_RUN_BEHAVIOR_TESTS=1. Booting Isaac takes minutes and can be
  done only once per process, so the whole tier shares one environment.
"""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import behavior as B
from embodiedscore_envs.benchmarks.env import BehaviorEnv, BehaviorPoseEnv, ManipGoal, targets_of
from embodiedscore_envs.benchmarks.env.behavior_env import ARM_ACTION, POSE_ACTION_DIM
from embodiedscore_envs.benchmarks.env.sim.behavior.body import ARMS, CAMERA_PRIMS
from embodiedscore_envs.benchmarks.env.sim.behavior.scene import SCENE_MODELS
from embodiedscore_envs.benchmarks.presets import bodies

LINES = sorted({b.line for b in es.BENCHMARKS.values() if b.engine == "behavior"})


def _has_metadata() -> bool:
    try:
        return (B.metadata_dir() / "test_instances.csv").exists() and (B.metadata_dir() / "episodes.jsonl").exists()
    except ValueError:
        return False


needs_metadata = pytest.mark.skipif(not _has_metadata(),
                                    reason="the 2025-challenge-task-instances metadata is not under "
                                           "OMNIGIBSON_DATA_PATH")


# ---- declarations (no data, no simulator) -------------------------------------------------

def test_declarations():
    assert LINES == ["behavior-1k"]
    std, up = es.benchmark("behavior-1k"), es.benchmark("behavior-1k", "upstream")
    assert std.engine == up.engine == "behavior" and std.line == up.line == "behavior-1k"
    assert std.macro and not up.macro
    assert std.body is bodies.BEHAVIOR_STANDARD and up.body is bodies.BEHAVIOR
    assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
    assert std.splits == up.splits == B.SPLITS == ("all", "mini")
    assert std.ticks is None and up.ticks is None       # the cap is the task's own budget
    # Both variants keep BEHAVIOR's own per-task tick cap; only the agent's budget unit bends.
    assert std.tick_scale == up.tick_scale == B.TICK_SCALE == 1.0
    assert std.max_episode_steps == B.MACRO_STEPS and up.max_episode_steps is None
    assert up.gym_id == "EmbodiedScore/BEHAVIOR-1K-Upstream-v0"
    assert std.gym_id == "EmbodiedScore/BEHAVIOR-1K-v0"


def test_the_fifty_challenge_tasks():
    """eval_utils.py TASK_NAMES_TO_INDICES L185-241 — 50 tasks, contiguous indices,
    each sampled into one of the release's three scene models."""
    assert len(B.TASKS) == 50 and len(set(B.TASK_INDEX)) == 50
    assert sorted(B.TASK_INDEX.values()) == list(range(50))
    assert B.TASK_INDEX["turning_on_radio"] == 0 and B.TASK_INDEX["make_pizza"] == 49
    assert set(B.SCENE_OF.values()) <= set(SCENE_MODELS)
    counts = {model: sum(1 for _t, m in B.TASKS if m == model) for model in SCENE_MODELS}
    # available_tasks.yaml, the 50 challenge tasks: 23 + 22 + 5
    assert counts == {"house_single_floor": 23, "house_double_floor_lower": 22, "house_double_floor_upper": 5}


def test_the_robot_is_the_challenges():
    body = bodies.BEHAVIOR
    assert body.robot == "R1Pro" and body.arms == ARMS == ("left", "right")
    # eval_utils.generate_basic_environment_config L256-261
    assert (body.action_freq, body.render_freq, body.physics_freq) == (30, 30, 120)
    assert body.head_aperture_mm == 40.0            # rgb_low_res_wrapper.py L24
    assert set(CAMERA_PRIMS) == {"head", "left_wrist", "right_wrist"}
    assert body.cameras == ("head", "left_wrist", "right_wrist")
    assert body.rgb == bodies.BEHAVIOR.wrist                     # RGBLowResWrapper renders all three at 224²
    assert (body.rgb.width, body.rgb.height) == (224, 224)
    assert (bodies.BEHAVIOR_STANDARD.rgb.width, bodies.BEHAVIOR_STANDARD.rgb.height) == (256, 256)
    assert body.depth is None and not body.has_tilt
    assert body.settle_ticks == 0                   # BEHAVIOR settles inside the instance load


def test_scene_ref_names_the_instance_file():
    scene = B.BehaviorSceneRef(task="turning_on_radio", scene_model="house_double_floor_lower", instance_id=242)
    # behavior_task.get_cached_activity_scene_filename L157-174
    assert scene.tro_stem == "house_double_floor_lower_task_turning_on_radio_0_242_template"
    assert scene.model_key == "house_double_floor_lower"
    assert scene.task_key == ("house_double_floor_lower", "turning_on_radio", 0)
    assert scene.scene_file == "house_double_floor_lower/turning_on_radio/0-242"


def test_manip_goal_has_no_place():
    g = ManipGoal((("toggled_on", "?radio_receiver.n.01_1"),))
    assert g.kind == "manip" and targets_of(g) is None


def test_pose_actions_encode_the_three_moves():
    hold = BehaviorPoseEnv.hold_action(+1, -1)
    assert hold.shape == (POSE_ACTION_DIM,) == (17,)
    assert not np.isfinite(hold[0:3]).any() and not np.isfinite(hold[7:10]).any()
    assert hold[6] == 1.0 and hold[13] == -1.0 and (hold[14:17] == 0).all()

    drive = BehaviorPoseEnv.base_action(0.5, 0.0, -0.25)
    assert not np.isfinite(drive[0:3]).any() and list(drive[14:17]) == [0.5, 0.0, -0.25]

    left = BehaviorPoseEnv.arm_action("left", [1.0, 2.0, 3.0], [0.0, 0.0, 0.1], +1)
    assert np.isfinite(left[0:3]).all() and not np.isfinite(left[7:10]).any()
    assert left[6] == 1.0 and (left[14:17] == 0).all()

    right = BehaviorPoseEnv.arm_action("right", [1.0, 2.0, 3.0], [0.0, 0.0, 0.1], +1, other_gripper=+1)
    assert np.isfinite(right[ARM_ACTION:ARM_ACTION + 3]).all() and not np.isfinite(right[0:3]).any()
    assert right[6] == 1.0 and right[13] == 1.0

    with pytest.raises(ValueError):
        BehaviorPoseEnv.arm_action("middle", [0, 0, 0], [0, 0, 0], -1)


def test_metric_keys_are_the_challenges():
    from embodiedscore_envs.benchmarks.env import BEHAVIOR_KEYS

    assert BEHAVIOR_KEYS[:2] == ("success", "q_score")       # the ranking metric, score_utils.py L114/L122
    for part in ("base", "left", "right"):
        assert f"agent_distance_{part}" in BEHAVIOR_KEYS
        assert f"normalized_agent_distance_{part}" in BEHAVIOR_KEYS
    assert "normalized_time" in BEHAVIOR_KEYS and "simulator_time_s" in BEHAVIOR_KEYS


# ---- the loader, against the release's own metadata ----------------------------------------

@needs_metadata
def test_public_test_instances_match_the_release_csv():
    instances = B.public_test_instances()
    assert len(instances) == 50
    for task, ids in instances.items():
        # the row holds 20 ids; the leaderboard scores the first 10 (eval.py L430-446)
        assert len(ids) == B.CSV_INSTANCES == 20, task
        assert len(set(ids)) == 20 and all(0 <= i <= 300 for i in ids), task
    assert instances["turning_on_radio"][:3] == [242, 295, 211]


@needs_metadata
def test_human_stats_are_the_two_hundred_demonstrations():
    stats = B.human_stats()
    assert len(stats) == 50
    for task, demo in stats.items():
        assert demo["length"] > 0 and demo["distance_traveled"] >= 0, task
        assert set(demo) == {"length", "distance_traveled", "left_eef_displacement", "right_eef_displacement"}


@needs_metadata
def test_episodes_are_the_challenges_five_hundred_rollouts():
    mini = B.load_episodes("mini")
    assert len(mini) == 50 == len(B.TASKS)                       # one public-test instance per task
    assert len(B.load_episodes("all")) == 50 * B.EVAL_INSTANCES == 500
    first = mini[0]
    assert first.scene.task == "turning_on_radio" and first.scene.scene_model == "house_double_floor_lower"
    assert first.scene.definition_id == 0 and first.scene.release == B.RELEASE
    assert first.episode_id == f"turning_on_radio/{first.scene.instance_id}"
    assert first.start_position is None and first.start_rotation is None
    assert first.goal.kind == "manip" and first.goal.predicates
    assert first.instruction == "turning on radio"
    # eval.py L146-153: the budget is twice the mean human demo length of that task
    assert first.info["max_steps"] == int(first.info["human_steps"] * B.BUDGET_MULTIPLE)
    assert set(first.info["human_distance"]) == {"base", "left", "right"}
    assert mini[1].scene.task != first.scene.task                # task-major
    every = B.load_episodes("all")
    assert every[0].scene.task == every[9].scene.task and every[10].scene.task != every[0].scene.task
    assert [e.index for e in every] == list(range(500))
    # the ten scored instances of a task are the CSV row's first ten, in its order
    row = B.public_test_instances()["turning_on_radio"]
    assert [e.scene.instance_id for e in every[:10]] == row[:B.EVAL_INSTANCES]


@needs_metadata
def test_instance_files_exist_for_the_mini_split():
    root = B.data_path()
    for ep in B.load_episodes("mini"):
        path = (root / B.INSTANCE_DIR / "scenes" / ep.scene.scene_model / "json"
                / f"{ep.scene.scene_model}_task_{ep.scene.task}_instances" / f"{ep.scene.tro_stem}-tro_state.json")
        assert path.exists(), f"{ep.episode_id}: no {path}"


def test_goal_predicates_come_from_bddl():
    """The activity's BDDL goal, parsed with no simulator and no backend
    (``bddl.parsing.parse_problem``)."""
    pytest.importorskip("bddl")
    goal = B.goal_predicates("turning_on_radio")
    assert goal == (("toggled_on", "?radio_receiver.n.01_1"),)
    trash = B.goal_predicates("picking_up_trash")
    assert len(trash) == 1 and "inside" in trash[0] and "ashcan.n.01_1" in " ".join(trash[0])
    for task, _scene in B.TASKS:
        assert B.goal_predicates(task), task


def test_every_challenge_task_has_a_bddl_definition():
    pytest.importorskip("bddl")
    from bddl.config import ACTIVITY_CONFIGS_PATH

    for task, _scene in B.TASKS:
        assert os.path.exists(os.path.join(ACTIVITY_CONFIGS_PATH, task, "problem0.bddl")), task


# ---- the simulator ---------------------------------------------------------------------------

def _skip_without_simulator() -> None:
    pytest.importorskip("omnigibson")
    if not os.environ.get("EMBODIEDSCORE_RUN_BEHAVIOR_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_BEHAVIOR_TESTS not set")
    if not _has_metadata():
        pytest.skip("the 2025-challenge-task-instances metadata is not under OMNIGIBSON_DATA_PATH")


@pytest.fixture(scope="module")
def stack():
    """One environment for the whole module: Isaac Sim boots once per process
    (``simulator.py`` L386) and a scene load costs minutes."""
    _skip_without_simulator()
    env = es.make("behavior-1k", "mini")
    yield env
    env.close()


def test_reset_and_facts(stack):
    base = stack.unwrapped
    assert isinstance(base, BehaviorPoseEnv) and isinstance(base, BehaviorEnv)
    obs, info = stack.reset(options={"episode": 0})
    body = bodies.BEHAVIOR_STANDARD
    assert set(obs) == {"rgb", "wrist", "wrist_right", "proprio"}
    assert obs["rgb"].shape == (body.rgb.height, body.rgb.width, 3)
    assert obs["wrist"].shape == (body.wrist.height, body.wrist.width, 3)
    assert obs["proprio"].shape == (18,)
    assert info["success"] is False and info["metrics"]["success"] == 0.0
    assert info["metrics"]["q_score"] == 0.0 and info["metrics"]["steps_taken"] == 0
    assert set(info["arms"]) == {"left", "right"}
    for arm in ARMS:
        assert len(info["arms"][arm]["eef_position"]) == 3 and len(info["arms"][arm]["eef_rotation"]) == 4
        assert 0.0 <= info["arms"][arm]["gripper_open"] <= 100.0
    assert len(info["base_position"]) == 3 and isinstance(info["base_yaw_rad"], float)
    assert len(info["trunk_qpos"]) == 4          # r1.py L208-213: four torso joints
    assert info["n_predicates"] >= 1 and set(info["goal_status"]) <= {"satisfied", "unsatisfied"}
    assert info["tick_cap"] == base.episode.info["max_steps"]
    assert info["language"]


def test_the_action_layout_is_the_challenges(stack):
    """The macro protocol substitutes IK for the two arms; everything else is
    the challenge's own controller set, in its own order."""
    world = stack.unwrapped.world
    slices = world.action_slices
    assert list(slices) == ["base", "trunk", "arm_left", "gripper_left", "arm_right", "gripper_right"]
    assert slices["base"].stop - slices["base"].start == 3
    assert slices["trunk"].stop - slices["trunk"].start == 4
    for arm in ARMS:
        assert slices[f"arm_{arm}"].stop - slices[f"arm_{arm}"].start == 6      # absolute_pose IK
        assert slices[f"gripper_{arm}"].stop - slices[f"gripper_{arm}"].start == 1
    assert world.action_dim == 21
    no_op = world.no_op_action()
    assert no_op.shape == (21,) and np.isfinite(no_op).all()


def test_gripper_hold_closes_and_opens(stack):
    obs, info = stack.reset(options={"episode": 0})
    obs, r, term, trunc, info = stack.step(BehaviorPoseEnv.hold_action(+1, +1))
    assert r == 0.0 and info["move_kind"] == "gripper" and info["converged"]
    closed = {arm: info["arms"][arm]["gripper_open"] for arm in ARMS}
    obs, r, term, trunc, info = stack.step(BehaviorPoseEnv.hold_action(-1, -1))
    for arm in ARMS:
        assert info["arms"][arm]["gripper_open"] > closed[arm] - 1e-6, arm
    assert info["metrics"]["steps_taken"] == 2


def test_arm_move_reduces_the_error(stack):
    obs, info = stack.reset(options={"episode": 0})
    from scipy.spatial.transform import Rotation

    pos = np.asarray(info["arms"]["left"]["eef_position"])
    w, x, y, z = info["arms"]["left"]["eef_rotation"]
    rotvec = Rotation.from_quat([x, y, z, w]).as_rotvec()
    target = pos + [0.0, 0.0, 0.05]
    obs, r, term, trunc, info = stack.step(BehaviorPoseEnv.arm_action("left", target, rotvec, -1.0))
    assert info["move_kind"] == "pose" and info["move_ticks"] > 0
    reached = np.asarray(info["arms"]["left"]["eef_position"])
    assert float(np.linalg.norm(target - reached)) < 0.05


def test_base_move_drives(stack):
    obs, info = stack.reset(options={"episode": 0})
    start = np.asarray(info["base_position"])[:2]
    obs, r, term, trunc, info = stack.step(BehaviorPoseEnv.base_action(-0.30, 0.0, 0.0))
    assert info["move_kind"] == "base"
    moved = float(np.linalg.norm(np.asarray(info["base_position"])[:2] - start))
    assert moved > 0.1
    assert info["metrics"]["agent_distance_base"] >= moved - 1e-6


def test_metrics_carry_the_human_reference(stack):
    obs, info = stack.reset(options={"episode": 0})
    obs, r, term, trunc, info = stack.step(BehaviorPoseEnv.hold_action(-1, -1))
    m = info["metrics"]
    # task_metric.gather_results: simulator_time = steps * rendering dt, 30 Hz here
    assert m["simulator_time_s"] == pytest.approx(m["ticks"] / 30.0)
    assert m["normalized_time"] == pytest.approx(stack.unwrapped.episode.info["human_steps"] / m["ticks"])
    for part in ("base", "left", "right"):
        assert m[f"agent_distance_{part}"] is not None
        assert m[f"normalized_agent_distance_{part}"] is not None


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env

    check_env(stack.unwrapped, skip_render_check=True)
