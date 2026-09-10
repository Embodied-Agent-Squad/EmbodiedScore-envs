"""The RoboCasa lines: the declaration contracts (no simulator), and the
Gymnasium env checker plus the shared facts on one episode (needs the
``robocasa`` package, its assets and EGL; opt in with
EMBODIEDSCORE_RUN_ROBOCASA_TESTS=1).

The simulator tests run against whichever release is installed in this
interpreter — RoboCasa365 (repo ``main``) or RoboCasa v0.2 — since the two
cannot share an environment (INSTALL-robocasa.md). ``EMBODIEDSCORE_ROBOCASA_LINE``
overrides the line they use.
"""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import robocasa as R
from embodiedscore_envs.benchmarks.env import ManipGoal, RobocasaEnv, RobocasaPoseEnv, targets_of
from embodiedscore_envs.benchmarks.env.robocasa_env import POSE_ACTION_DIM
from embodiedscore_envs.benchmarks.env.sim.robocasa.world import ACTION_DIM
from embodiedscore_envs.benchmarks.presets import bodies

LINES = sorted({b.line for b in es.BENCHMARKS.values() if b.engine == "robocasa"})
R365_LINES = [n for n in LINES if n.startswith("robocasa365-")]
R02_LINES = [n for n in LINES if not n.startswith("robocasa365-")]


def _installed_release() -> str | None:
    try:
        import robocasa
    except ImportError:
        return None
    return "robocasa365" if int(str(robocasa.__version__).split(".")[0]) >= 1 else "robocasa"


def _line_for_release() -> str:
    override = os.environ.get("EMBODIEDSCORE_ROBOCASA_LINE")
    if override:
        return override
    return "robocasa365-atomic-seen" if _installed_release() == "robocasa365" else "robocasa-drawers"


def test_declarations():
    assert R365_LINES == ["robocasa365-atomic-seen", "robocasa365-composite-seen", "robocasa365-composite-unseen"]
    assert R02_LINES == ["robocasa-buttons", "robocasa-doors", "robocasa-drawers", "robocasa-insertion",
                         "robocasa-knobs", "robocasa-levers", "robocasa-navigate", "robocasa-pnp"]
    for line in LINES:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert std.engine == up.engine == "robocasa" and std.line == up.line == line
        assert std.macro and not up.macro
        assert std.body is bodies.ROBOCASA_STANDARD and up.body is bodies.ROBOCASA
        assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
        assert std.splits == up.splits == R.SPLITS
        assert std.ticks is None and up.ticks is None       # the cap is the task's own horizon
        assert up.tick_scale == 1.0 and std.tick_scale == float(R.TICK_GUARD)
        assert up.max_episode_steps == max(R.LINES[line]["tasks"].values())
        assert std.max_episode_steps == R.MACRO_STEPS
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")


def test_task_counts_match_the_benchmarks_own_grouping():
    # RoboCasa365 TASK_SET_REGISTRY: the three sets the leaderboard scores (18 + 16 + 16 = target50)
    assert [len(R.LINES[n]["tasks"]) for n in R365_LINES] == [18, 16, 16]
    assert sum(len(R.LINES[n]["tasks"]) for n in R365_LINES) == 50
    # RoboCasa v0.2: the paper's eight foundational skills over 25 atomic tasks
    assert sum(len(R.LINES[n]["tasks"]) for n in R02_LINES) == 25
    assert len(R02_LINES) == 8


def test_episodes_are_the_evaluation_protocols_scenarios():
    for line, scenes in ((R365_LINES[0], R.R365_SCENES), (R02_LINES[0], R.R02_SCENES)):
        tasks = R.LINES[line]["tasks"]
        assert R.scenes_of(next(iter(tasks)), R.RELEASES[R.LINES[line]["release"]]) == scenes
        eps = R.load_episodes(line, "mini")
        assert len(eps) == len(tasks) * R.MINI_SCENARIOS
        assert len(R.load_episodes(line, "all")) == len(tasks) * R.SCENARIOS
        first = eps[0]
        assert first.scene.task == next(iter(tasks))
        assert (first.scene.layout_id, first.scene.style_id) == scenes[0]
        assert first.scene.seed == 0 and first.goal.kind == "manip"
        assert first.info["horizon"] == tasks[first.scene.task]
        # scenarios are dealt round-robin over the line's evaluation scenes
        k = min(len(scenes), R.MINI_SCENARIOS - 1)
        assert (eps[k].scene.layout_id, eps[k].scene.style_id) == scenes[k % len(scenes)]
        assert eps[R.MINI_SCENARIOS].scene.task != first.scene.task   # task-major


def test_excluded_scenes_are_never_dealt():
    for line, decl in R.LINES.items():
        release = R.RELEASES[decl["release"]]
        for ep in R.load_episodes(line, "all"):
            ruled_out = release["excluded"].get(ep.scene.task, ())
            assert (ep.scene.layout_id, ep.scene.style_id) not in ruled_out
    # nine of RoboCasa365's composite tasks rule at least one target scene out
    assert len(R.R365_EXCLUDED_SCENES) == 9 and R.R02_EXCLUDED_SCENES == {}
    assert R.scenes_of("DeliverStraw", R.RELEASES["robocasa365"]) == ((2, 2), (4, 4), (7, 7), (8, 8), (9, 9), (10, 10))


def test_manip_goal_has_no_place():
    g = ManipGoal((("robocasa_check_success", "OpenDrawer"),))
    assert g.kind == "manip" and targets_of(g) is None


def test_pose_actions_encode_the_three_moves():
    hold = RobocasaPoseEnv.hold_action(+1)
    assert hold.shape == (POSE_ACTION_DIM,) and not np.isfinite(hold[:3]).any() and hold[6] == 1.0
    assert (hold[7:10] == 0).all()
    drive = RobocasaPoseEnv.base_action(0.5, 0.0, -0.25, gripper=-1)
    assert not np.isfinite(drive[:3]).any() and list(drive[7:10]) == [0.5, 0.0, -0.25]
    reach = RobocasaPoseEnv.pose_action([1.0, 2.0, 3.0], [0.0, 0.0, 0.1], +1)
    assert np.isfinite(reach[:3]).all() and (reach[7:10] == 0).all()


# ---- the simulator ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack():
    pytest.importorskip("robocasa")
    if not os.environ.get("EMBODIEDSCORE_RUN_ROBOCASA_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_ROBOCASA_TESTS not set")
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = es.make(_line_for_release(), "mini")
    yield env
    env.close()


def test_tables_match_the_installed_release():
    pytest.importorskip("robocasa")
    if not os.environ.get("EMBODIEDSCORE_RUN_ROBOCASA_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_ROBOCASA_TESTS not set")
    release = _installed_release()
    horizons = R.installed_horizons(release)
    for line, decl in R.LINES.items():
        if decl["release"] != release:
            continue
        for task, horizon in decl["tasks"].items():
            assert horizons.get(task) == horizon, f"{line}/{task}: table {horizon}, installed {horizons.get(task)}"


def test_excluded_scenes_match_the_installed_release():
    pytest.importorskip("robocasa")
    if not os.environ.get("EMBODIEDSCORE_RUN_ROBOCASA_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_ROBOCASA_TESTS not set")
    release = _installed_release()
    table = R.RELEASES[release]["excluded"]
    assert R.installed_exclusions(release) == table


def test_reset_and_facts(stack):
    base = stack.unwrapped
    assert isinstance(base, RobocasaPoseEnv) and isinstance(base, RobocasaEnv)
    obs, info = stack.reset(options={"episode": 0})
    body = bodies.ROBOCASA_STANDARD
    assert set(obs) == {"rgb", "wrist", "proprio"}
    assert obs["rgb"].shape == (body.rgb.height, body.rgb.width, 3) and obs["proprio"].shape == (12,)
    assert info["metrics"] == {"success": 0.0, "steps_taken": 0, "ticks": body.settle_ticks}
    assert info["success"] is False and len(info["eef_position"]) == 3 and len(info["eef_rotation"]) == 4
    assert len(info["base_position"]) == 3 and isinstance(info["base_yaw_rad"], float)
    assert info["language"] and info["episode"]["instruction"] == info["language"]
    assert info["tick_cap"] == R.TICK_GUARD * base.episode.info["horizon"]


def test_pose_move_converges(stack):
    obs, info = stack.reset(options={"episode": 0})
    from scipy.spatial.transform import Rotation
    pos = np.asarray(info["eef_position"])
    w, x, y, z = info["eef_rotation"]
    rotvec = Rotation.from_quat([x, y, z, w]).as_rotvec()
    obs, r, term, trunc, info = stack.step(RobocasaPoseEnv.pose_action(pos + [0.0, 0.0, -0.05], rotvec, -1.0))
    assert r == 0.0 and not term and info["move_kind"] == "pose"
    assert info["converged"] and info["position_error_m"] < 0.011
    assert info["metrics"]["steps_taken"] == 1
    obs, r, term, trunc, info = stack.step(RobocasaPoseEnv.hold_action(+1))
    assert info["move_kind"] == "gripper" and info["gripper_open"] < 20
    assert info["metrics"]["steps_taken"] == 2


def test_base_move_converges(stack):
    obs, info = stack.reset(options={"episode": 0})
    start = np.asarray(info["base_position"])[:2]
    obs, r, term, trunc, info = stack.step(RobocasaPoseEnv.base_action(-0.30, 0.0, 0.0))
    assert info["move_kind"] == "base" and info["converged"]
    moved = float(np.linalg.norm(np.asarray(info["base_position"])[:2] - start))
    assert 0.2 < moved < 0.4


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env
    check_env(stack.unwrapped, skip_render_check=True)


def test_upstream_protocol_is_robocasas_own_tick():
    pytest.importorskip("robocasa")
    if not os.environ.get("EMBODIEDSCORE_RUN_ROBOCASA_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_ROBOCASA_TESTS not set")
    env = es.make(_line_for_release() + "-upstream", "mini")
    try:
        obs, info = env.reset(options={"episode": 0})
        body = bodies.ROBOCASA
        assert set(obs) == {"rgb", "wrist", "aux", "proprio"}
        assert obs["rgb"].shape == (body.rgb.height, body.rgb.width, 3)
        assert isinstance(env.unwrapped, RobocasaEnv) and not isinstance(env.unwrapped, RobocasaPoseEnv)
        assert env.unwrapped.action_space.shape == (ACTION_DIM,)
        assert info["tick_cap"] == env.unwrapped.episode.info["horizon"]
        for n in range(3):
            obs, r, term, trunc, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            assert not trunc and info["metrics"]["steps_taken"] == n + 1
    finally:
        env.close()
