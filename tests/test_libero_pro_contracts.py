"""The LIBERO-PRO lines: the declaration contracts and the pure loader helpers
(no simulator), then the episodes and one macro episode on the real thing
(needs the LIBERO-PRO fork installed as ``libero`` and EGL; opt in with
EMBODIEDSCORE_RUN_LIBERO_PRO_TESTS=1)."""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import libero_pro as P
from embodiedscore_envs.benchmarks.env import LiberoEnv, LiberoPoseEnv
from embodiedscore_envs.benchmarks.presets import bodies

LINES = sorted({b.line for b in es.BENCHMARKS.values() if b.name.startswith("libero-pro-")})
SUITE_OF = {f"libero-pro-{word}": suite for word, suite in P.SUITES.items()}


def test_declarations():
    assert LINES == ["libero-pro-10", "libero-pro-goal", "libero-pro-object", "libero-pro-spatial"]
    for line in LINES:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert std.engine == up.engine == "libero" and std.line == up.line == line
        assert std.macro and not up.macro
        assert std.body is bodies.LIBERO_STANDARD and up.body is bodies.LIBERO
        assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
        ticks = P.TICKS[SUITE_OF[line]]
        assert up.max_episode_steps == up.ticks == ticks
        assert std.max_episode_steps == P.MACRO_STEPS and std.ticks == ticks * P.TICK_GUARD
        assert std.splits == up.splits == P.SPLITS
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")


def test_the_four_released_dimensions():
    """The dimensions are the four LIBERO-PRO releases task files for; the
    tick cap is the base suite's, as its README's TASK_MAX_STEPS repeats it."""
    assert [word for word, _, _ in P.DIMENSIONS] == ["object", "position", "semantic", "task"]
    assert [suffix for _, suffix, _ in P.DIMENSIONS] == ["object", "swap", "lan", "task"]
    assert P.TICKS == {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def test_a_line_without_its_own_tick_cap_uses_the_declarations():
    """LIBERO-PRO lines draw from one suite, so no episode overrides the cap."""
    from embodiedscore_envs.benchmarks.env.schema import Episode, LiberoSceneRef, ManipGoal
    scene = LiberoSceneRef(suite="libero_spatial_swap", task_id=0, bddl_file="a.bddl", init_states_file="a.pruned_init")
    ep = Episode(index=0, episode_id="libero_spatial_swap/0/0", scene=scene, start_position=None,
                 start_rotation=None, goal=ManipGoal(()), info={"init_state_index": 0})
    assert LiberoEnv.episode_ticks(ep, 220) == 220
    assert LiberoEnv.episode_ticks(ep, None) is None


@pytest.fixture(scope="module")
def stack():
    pytest.importorskip("libero")
    if not os.environ.get("EMBODIEDSCORE_RUN_LIBERO_PRO_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_LIBERO_PRO_TESTS not set")
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = es.make("libero-pro-spatial", "mini")
    yield env
    env.close()


def test_episodes_are_dimension_major(stack):
    base = stack.unwrapped
    assert isinstance(base, LiberoPoseEnv)
    tasks_per_dim = 10
    assert len(base.episodes) == len(P.DIMENSIONS) * tasks_per_dim * P.MINI_INIT_STATES
    first = base.episodes[0]
    assert first.episode_id == "libero_spatial_object/0/0" and first.info["perturbation"] == "object"
    assert first.goal.kind == "manip" and first.start_position is None
    block = tasks_per_dim * P.MINI_INIT_STATES
    assert base.episodes[block].info["perturbation"] == "position"
    assert base.episodes[block].episode_id == "libero_spatial_swap/0/0"
    assert base.episodes[2 * block].info["perturbation"] == "semantic"
    assert base.episodes[3 * block].info["perturbation"] == "task"
    assert {e.info["base_suite"] for e in base.episodes} == {"libero_spatial"}
    assert all("max_ticks" not in e.info for e in base.episodes)


def test_the_instruction_comes_from_the_bddl_not_the_registry(stack):
    """The fork derives ``task.language`` from the filename, which the
    perturbation does not change; the semantic dimension only exists in the
    BDDL's own (:language ...) field."""
    from libero.libero import benchmark as libero_benchmark

    base = stack.unwrapped
    block = 10 * P.MINI_INIT_STATES
    semantic = base.episodes[2 * block]
    registry = libero_benchmark.get_benchmark_dict()["libero_spatial_lan"]().get_task(0)
    assert semantic.instruction and semantic.instruction != str(registry.language)
    unperturbed = base.episodes[0].instruction          # the object dimension keeps the base sentence
    assert unperturbed and unperturbed != semantic.instruction


def test_reset_step_and_facts(stack):
    obs, info = stack.reset(options={"episode": 0})
    assert set(obs) == {"rgb", "wrist", "proprio"} and obs["rgb"].shape == (256, 256, 3)
    assert info["metrics"] == {"success": 0.0, "steps_taken": 0, "ticks": bodies.LIBERO_STANDARD.settle_ticks}
    assert info["success"] is False and info["objects"] and info["gripper_open"] > 90
    pos = np.asarray(info["eef_position"])
    from scipy.spatial.transform import Rotation
    w, x, y, z = info["eef_rotation"]
    rot = Rotation.from_quat([x, y, z, w]).as_rotvec()
    obs, r, term, trunc, info = stack.step(np.concatenate([pos + [0.0, 0.0, -0.05], rot, [-1.0]]))
    assert r == 0.0 and not term and not trunc and info["converged"] and info["metrics"]["steps_taken"] == 1


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env
    check_env(stack.unwrapped, skip_render_check=True)


def test_a_scripted_episode_succeeds(stack):
    """The proof the macro protocol can close a perturbed task — the same
    privileged-info routine as scripts/libero_scripted_episode.py."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from libero_scripted_episode import run_episode

    success, metrics = run_episode("libero-pro-spatial", "mini", 0, verbose=False)
    assert success == 1.0 and metrics["success"] == 1.0
