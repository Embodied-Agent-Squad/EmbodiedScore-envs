"""The LIBERO-Plus lines: the declaration contracts and the pure name / path
helpers (no simulator), then the episodes and one macro episode on the real
thing (needs LIBERO-Plus installed as ``libero`` and EGL; opt in with
EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS=1)."""

import os

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import libero_plus as L
from embodiedscore_envs.benchmarks.env import LiberoEnv, LiberoPoseEnv
from embodiedscore_envs.benchmarks.presets import bodies

LINES = sorted({b.line for b in es.BENCHMARKS.values() if b.name.startswith("libero-plus-")})
# The released task count per category — task_classification.json, and Table 7 of arXiv:2510.13626.
TASKS = {"camera": 1599, "noise": 1601, "robot": 1550, "language": 1537,
         "layout": 1525, "light": 1142, "background": 1076}


def test_declarations():
    assert LINES == sorted(f"libero-plus-{w}" for w in L.LINES)
    longest = max(L.TICKS.values())
    for line in LINES:
        std, up = es.benchmark(line), es.benchmark(line, "upstream")
        assert std.engine == up.engine == "libero" and std.line == up.line == line
        assert std.macro and not up.macro
        assert std.body is bodies.LIBERO_STANDARD and up.body is bodies.LIBERO
        assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
        assert up.max_episode_steps == up.ticks == longest
        assert std.max_episode_steps == L.MACRO_STEPS and std.ticks == longest * L.TICK_GUARD
        assert std.splits == up.splits == L.SPLITS
        assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")


def test_the_seven_categories_and_their_counts():
    assert set(L.LINES.values()) == {"Camera Viewpoints", "Sensor Noise", "Robot Initial States",
                                     "Language Instructions", "Objects Layout", "Light Conditions",
                                     "Background Textures"}
    assert sum(TASKS.values()) == 10030 and set(TASKS) == set(L.LINES)
    assert L.BASE_SUITES == ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    assert L.INIT_STATE == 0     # num_trials_per_task = 1


def test_base_task_name_strips_every_suffix():
    base = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    for name in (base,
                 f"{base}_view_0_0_100_2_352_initstate_0",
                 f"{base}_view_0_0_100_0_0_initstate_141",
                 f"{base}_view_0_0_100_0_0_initstate_0_noise_7",
                 f"{base}_language_1_view_0_0_100_0_0_initstate_0",
                 f"{base}_table_13", f"{base}_tb_4", f"{base}_light_1", f"{base}_add_10",
                 f"{base}_level5_sample4", f"{base}_moved"):
        assert L._base_task_name(name) == base, name


def test_path_resolution_follows_the_fork():
    """``_resolve_bddl`` / ``_resolve_init`` transcribe env_wrapper.ControlEnv
    and Benchmark.get_task_init_states — the four encoded categories share the
    base task's files, the layout ones have their own under libero_newobj."""
    base = "put_the_bowl_on_the_plate"
    b = L._resolve_bddl("/b", "libero_goal", f"{base}_view_0_0_100_2_352_initstate_0.bddl")
    assert b == f"/b/libero_goal/{base}.bddl"
    assert L._resolve_bddl("/b", "libero_goal", f"{base}_light_1.bddl") == f"/b/libero_goal/{base}_light_1.bddl"
    i = L._resolve_init("/i", "libero_goal", f"{base}_language_2_view_0_0_100_0_0_initstate_0.pruned_init")
    assert i == f"/i/libero_goal/{base}.pruned_init"
    assert L._resolve_init("/i", "libero_goal", f"{base}_table_13.pruned_init") == f"/i/libero_goal/{base}.pruned_init"
    assert L._resolve_init("/i", "libero_goal", f"{base}_tb_4.pruned_init") == f"/i/libero_goal/{base}.pruned_init"
    assert L._resolve_init("/i", "libero_goal", f"{base}_add_10.pruned_init") == \
        f"/i/libero_newobj/libero_goal/{base}_add_10.pruned_init"
    assert L._resolve_init("/i", "libero_goal", f"{base}.pruned_init") == f"/i/libero_goal/{base}.pruned_init"


def test_a_mixed_suite_line_carries_its_cap_per_episode():
    from embodiedscore_envs.benchmarks.env.schema import Episode, LiberoSceneRef, ManipGoal
    scene = LiberoSceneRef(suite="libero_10", task_id=0, bddl_file="a.bddl", init_states_file="a.pruned_init")
    ep = Episode(index=0, episode_id="libero_10/0/0", scene=scene, start_position=None, start_rotation=None,
                 goal=ManipGoal(()), info={"init_state_index": 0, "max_ticks": 520})
    assert LiberoEnv.episode_ticks(ep, 220) == 520


@pytest.fixture(scope="module")
def stack():
    pytest.importorskip("libero")
    if not os.environ.get("EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS not set")
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = es.make("libero-plus-camera", "mini")
    yield env
    env.close()


def test_mini_split_is_ten_tasks_per_base_suite(stack):
    base = stack.unwrapped
    assert isinstance(base, LiberoPoseEnv)
    assert len(base.episodes) == len(L.BASE_SUITES) * L.MINI_TASKS_PER_SUITE
    suites = [e.info["base_suite"] for e in base.episodes]
    assert suites[:1] == ["libero_spatial"] and set(suites) == set(L.BASE_SUITES)
    assert suites == sorted(suites, key=L.BASE_SUITES.index)          # suite-major
    for ep in base.episodes:
        assert ep.info["category"] == "Camera Viewpoints" and ep.info["perturbation"] == "camera"
        assert ep.info["max_ticks"] == L.TICKS[ep.info["base_suite"]] * L.TICK_GUARD
        assert ep.episode_id.endswith("/0") and ep.goal.kind == "manip"
        assert "_view_" in ep.info["task_name"] and "_view_" not in ep.instruction


def test_the_instruction_drops_the_encoded_suffix(stack):
    from libero.libero import benchmark as libero_benchmark

    ep = stack.unwrapped.episodes[0]
    registry = libero_benchmark.get_benchmark_dict()[ep.info["base_suite"]]().get_task(ep.info["task_id"])
    assert "initstate" in str(registry.language)                      # what the fork would have handed the agent
    assert "initstate" not in ep.instruction and "view" not in ep.instruction.split()
    assert ep.instruction == " ".join(ep.info["base_task_name"].split("_"))


def test_every_line_loads_its_released_task_count():
    pytest.importorskip("libero")
    if not os.environ.get("EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS not set")
    for word, expected in TASKS.items():
        episodes = L.load_episodes(word, "all")
        assert len(episodes) == expected, f"{word}: {len(episodes)} != {expected}"


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

    success, metrics = run_episode("libero-plus-camera", "mini", 0, verbose=False)
    assert success == 1.0 and metrics["success"] == 1.0
