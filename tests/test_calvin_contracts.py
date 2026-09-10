"""The CALVIN line: the declaration contracts (no simulator), the tables
checked against the upstream checkout when one is there, and the Gymnasium
env checker plus the chain protocol on one episode (needs ``calvin_env``,
the dataset config and EGL; opt in with EMBODIEDSCORE_RUN_CALVIN_TESTS=1).

The offline half needs nothing but the package. Generating the 1000
evaluation sequences takes about two minutes, so the tests that need them
use ``mini`` and the on-disk cache under ``EMBODIEDSCORE_DATA_ROOT``.
"""

import os
import sys

import numpy as np
import pytest

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks import calvin as C
from embodiedscore_envs.benchmarks.env import CalvinEnv, CalvinPoseEnv, GoalSequence, ManipGoal, targets_of
from embodiedscore_envs.benchmarks.env.calvin_env import ACTION_DIM
from embodiedscore_envs.benchmarks.presets import bodies

LINE = "calvin-d"


def _data_root() -> str | None:
    return os.environ.get("EMBODIEDSCORE_DATA_ROOT")


def _checkout() -> str | None:
    """The mees/calvin clone, found through the installed (editable)
    ``calvin_env``: ``<clone>/calvin_env/calvin_env/__init__.py``."""
    try:
        import calvin_env
    except ImportError:
        return None
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(calvin_env.__file__))))
    return repo if os.path.isdir(os.path.join(repo, "calvin_models")) else None


# ---- declarations (no simulator, no data) -------------------------------------------------

def test_declarations():
    std, up = es.benchmark(LINE), es.benchmark(LINE, "upstream")
    assert std.engine == up.engine == "calvin" and std.line == up.line == LINE
    assert std.macro and not up.macro
    assert std.body is bodies.CALVIN_STANDARD and up.body is bodies.CALVIN
    assert std.depth is None and up.depth is None and std.actions == () and up.actions == ()
    assert std.splits == up.splits == C.SPLITS == ("all", "mini")
    # the budget is PER SUB-TASK on both variants; max_episode_steps is only the episode ceiling
    assert up.ticks == C.EP_LEN == 360
    assert std.ticks == C.EP_LEN * C.TICK_GUARD
    assert up.max_episode_steps == C.EP_LEN * C.SEQ_LEN
    assert std.max_episode_steps == C.MACRO_STEPS == C.MACRO_STEPS_PER_SUBTASK * C.SEQ_LEN == 100
    assert up.gym_id.endswith("-Upstream-v0") and std.gym_id == up.gym_id.replace("-Upstream", "")


def test_one_line_because_both_board_rows_evaluate_in_D():
    """ABC->D and ABCD->D differ only in the training split; the evaluation
    environment and the 1000 sequences are the same (see calvin.py)."""
    assert sorted({b.line for b in es.BENCHMARKS.values() if b.engine == "calvin"}) == [LINE]
    assert C.ENV_LETTER == "D"


def test_the_task_table_is_complete_and_in_the_draw_order():
    assert len(C.TASKS) == 34 == len(set(C.TASKS))
    assert set(C.TASKS) == set(C.CATEGORIES) == set(C.INSTRUCTIONS) == set(C.RULES)
    assert tuple(C.RULES) == C.TASKS          # np.random.choice indexes this order
    assert len(set(C.CATEGORIES.values())) == 11
    assert all(C.INSTRUCTIONS[t].strip() for t in C.TASKS)


def test_initial_conditions_are_the_filtered_product():
    conditions = C.initial_conditions()
    assert len(conditions) == 192
    for c in conditions:
        assert list(c) == list(C.POSSIBLE_CONDITIONS)
        blocks = [c["red_block"], c["blue_block"], c["pink_block"]]
        assert blocks.count("table") in (1, 2)
        assert blocks.count("slider_right") < 2 and blocks.count("slider_left") < 2
        assert c["grasped"] == 0


def test_fnv1_32_is_pyhashs_default_seeded_variant():
    """pyhash's ``fnv1_32()`` starts from seed 0 (not the FNV-1 offset basis)
    and hashes a str as UTF-16 — both transcribed in calvin.py."""
    assert C.fnv1_32("") == 0
    # multiply-then-xor from 0 over UTF-16-LE bytes: "a" -> (0*prime)^0x61 = 0x61, then ^0x00
    assert C.fnv1_32("a") == ((0x61 * 0x01000193) & 0xFFFFFFFF) ^ 0x00
    assert 0 <= C.fnv1_32("dict_values([0, 1])") < 2 ** 32


def test_env_state_has_calvins_shapes_and_is_deterministic():
    condition = C.initial_conditions()[0]
    robot, scene = C.env_state_for(condition)
    assert len(robot) == 15 and len(scene) == 24        # euler_obs robot_obs, and the 24-D scene_obs
    assert C.env_state_for(dict(condition)) == (robot, scene)
    # the symbolic condition is realised in the state
    open_drawer = dict(condition, drawer="open")
    assert C.env_state_for(open_drawer)[1][1] == pytest.approx(0.22)
    assert C.env_state_for(dict(condition, drawer="closed"))[1][1] == 0.0
    assert C.env_state_for(dict(condition, led=1))[1][5] == 1.0


def test_pose_actions_encode_the_two_moves():
    hold = CalvinPoseEnv.hold_action(+1)
    assert hold.shape == (ACTION_DIM,) and not np.isfinite(hold[:3]).any() and hold[6] == 1.0
    reach = CalvinPoseEnv.pose_action([0.1, -0.2, 0.6], [0.0, 0.0, 0.1], -1)
    assert np.isfinite(reach[:3]).all() and reach[6] == -1.0


def test_the_chain_is_a_goal_sequence_of_manip_goals():
    goal = GoalSequence(tuple(ManipGoal((("calvin_task", t),)) for t in C.TASKS[:5]))
    assert goal.kind == "sequence" and len(goal.goals) == 5
    assert all(g.kind == "manip" and targets_of(g) is None for g in goal.goals)
    with pytest.raises(TypeError):
        targets_of(goal)


# ---- the tables against the upstream checkout ---------------------------------------------

def test_instructions_match_the_checkout():
    repo = _checkout()
    if repo is None:
        pytest.skip("the mees/calvin checkout is not next to the installed calvin_env")
    from omegaconf import OmegaConf

    path = os.path.join(repo, "calvin_models", "conf", "annotations", "new_playtable_validation.yaml")
    if not os.path.isfile(path):
        pytest.skip(f"no annotations at {path}")
    upstream = OmegaConf.to_container(OmegaConf.load(path))
    assert {k: v[0] for k, v in upstream.items()} == C.INSTRUCTIONS


def test_sequence_tables_match_the_checkout():
    """``multistep_sequences.tasks`` / ``task_categories``, which decide which
    chains exist and in what order they are drawn."""
    repo = _checkout()
    if repo is None:
        pytest.skip("the mees/calvin checkout is not next to the installed calvin_env")
    module = _import_multistep(repo)
    if module is None:
        pytest.skip("calvin_agent.evaluation.multistep_sequences is not importable")
    assert tuple(module.tasks.keys()) == C.TASKS
    assert module.task_categories == C.CATEGORIES

    def norm(rule):     # key order inside a condition dict is not semantic (_check_condition iterates it)
        return sorted(
            repr((sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in r["condition"].items()),
                  sorted(r["effect"].items())))
            for r in rule)

    for name in module.tasks:
        assert norm(module.tasks[name]) == norm(C.RULES[name]), name


def test_task_oracle_table_matches_the_installed_calvin_env():
    pytest.importorskip("calvin_env")
    from embodiedscore_envs.benchmarks.env.sim.calvin.world import task_oracle

    oracle = task_oracle()
    assert tuple(oracle.tasks.keys()) == C.TASKS
    assert oracle.num_tasks == 34


def _import_multistep(repo: str):
    """``calvin_agent.evaluation.multistep_sequences`` without dragging in
    torch. The only name it takes from ``calvin_agent.evaluation.utils`` is
    ``temp_seed``, and that module imports the MCIL model (torch +
    pytorch-lightning), which these lines deliberately do not install — so
    the module is stubbed with our own identical ``_temp_seed``."""
    import types

    models = os.path.join(repo, "calvin_models")
    if models not in sys.path:
        sys.path.insert(0, models)
    try:
        import calvin_agent.evaluation  # noqa: F401
    except ImportError:
        return None
    if "calvin_agent.evaluation.utils" not in sys.modules:
        stub = types.ModuleType("calvin_agent.evaluation.utils")
        stub.temp_seed = C._temp_seed
        sys.modules["calvin_agent.evaluation.utils"] = stub
    try:
        from calvin_agent.evaluation import multistep_sequences
    except ImportError:
        return None
    return multistep_sequences


def test_sequences_match_upstreams_generator():
    """The 1000 pairs, against upstream's own ``get_sequences`` (~2 min each
    side). Opt in with EMBODIEDSCORE_RUN_CALVIN_SEQ_TEST=1."""
    if not os.environ.get("EMBODIEDSCORE_RUN_CALVIN_SEQ_TEST"):
        pytest.skip("EMBODIEDSCORE_RUN_CALVIN_SEQ_TEST not set (the check takes minutes)")
    repo = _checkout()
    module = _import_multistep(repo) if repo else None
    if module is None:
        pytest.skip("the mees/calvin checkout is not next to the installed calvin_env")
    mine = C._generate(C.NUM_SEQUENCES)
    theirs = module.get_sequences(C.NUM_SEQUENCES)
    assert len(mine) == len(theirs) == C.NUM_SEQUENCES
    assert all(a[0] == b[0] and tuple(a[1]) == tuple(b[1]) for a, b in zip(mine, theirs))


# ---- the loader (needs the dataset config on disk, not the simulator) ----------------------

@pytest.fixture(scope="module")
def episodes():
    if not _data_root():
        pytest.skip("EMBODIEDSCORE_DATA_ROOT not set")
    return C.load_episodes("mini")


def test_episodes_are_the_evaluators_chains(episodes):
    assert len(episodes) == C.MINI_SEQUENCES == 100
    first = episodes[0]
    assert first.episode_id == "D/0" and first.index == 0
    assert first.scene.env_letter == "D" and first.scene.config_file.endswith("merged_config.yaml")
    assert first.start_position is None and first.start_rotation is None
    assert first.goal.kind == "sequence" and len(first.goal.goals) == C.SEQ_LEN == 5
    assert [g.predicates[0][1] for g in first.goal.goals] == first.info["sequence"]
    assert first.info["instructions"] == [C.INSTRUCTIONS[t] for t in first.info["sequence"]]
    assert first.instruction == first.info["instructions"][0]
    assert len(first.info["robot_obs"]) == 15 and len(first.info["scene_obs"]) == 24
    assert set(first.info["initial_condition"]) == set(C.POSSIBLE_CONDITIONS)
    # every chain is five distinct tasks in five distinct categories (check_sequence's rule)
    for ep in episodes:
        seq = ep.info["sequence"]
        assert len(seq) == 5 == len(set(seq))
        assert len({C.CATEGORIES[t] for t in seq}) == 5


def test_mini_is_the_prefix_of_all(episodes):
    all_eps = C.load_episodes("all")
    assert len(all_eps) == C.NUM_SEQUENCES == 1000
    assert [e.info["sequence"] for e in all_eps[:100]] == [e.info["sequence"] for e in episodes]


# ---- the simulator ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack():
    pytest.importorskip("calvin_env")
    if not os.environ.get("EMBODIEDSCORE_RUN_CALVIN_TESTS"):
        pytest.skip("EMBODIEDSCORE_RUN_CALVIN_TESTS not set")
    if not _data_root():
        pytest.skip("EMBODIEDSCORE_DATA_ROOT not set")
    env = es.make(LINE, "mini")
    yield env
    env.close()


def test_reset_and_facts(stack):
    base = stack.unwrapped
    assert isinstance(base, CalvinPoseEnv) and isinstance(base, CalvinEnv)
    obs, info = stack.reset(options={"episode": 0})
    body = bodies.CALVIN_STANDARD
    assert set(obs) == {"rgb", "wrist", "proprio"}
    assert obs["rgb"].shape == (body.rgb.height, body.rgb.width, 3)
    assert obs["wrist"].shape == (body.wrist.height, body.wrist.width, 3)
    assert obs["proprio"].shape == (15,)
    assert info["metrics"]["chain_length"] == 0 and info["metrics"]["success"] == 0.0
    assert info["metrics"]["n_subtasks"] == 5 and info["metrics"]["steps_taken"] == 0
    assert len(info["eef_position"]) == 3 and len(info["eef_rotation"]) == 4
    assert 0.0 <= info["gripper_open"] <= 100.0
    # the chain reveals ONE instruction
    assert info["goal_index"] == 0 and info["n_goals"] == 5
    assert info["instruction"] == base.episode.info["instructions"][0]
    assert info["subtask"] == base.episode.info["sequence"][0]
    assert info["subtask_closed"] is False and info["subtask_tick_cap"] == C.EP_LEN * C.TICK_GUARD
    assert info["subtask_move_cap"] == C.MACRO_STEPS_PER_SUBTASK


def test_pose_move_converges(stack):
    from scipy.spatial.transform import Rotation

    obs, info = stack.reset(options={"episode": 0})
    pos = np.asarray(info["eef_position"])
    w, x, y, z = info["eef_rotation"]
    rotvec = Rotation.from_quat([x, y, z, w]).as_rotvec()
    obs, r, term, trunc, info = stack.step(CalvinPoseEnv.pose_action(pos + [0.0, 0.0, -0.05], rotvec, -1.0))
    assert r == 0.0 and not term
    assert info["converged"] and info["position_error_m"] < 0.011
    assert info["metrics"]["steps_taken"] == 1 and info["subtask_moves"] == 1
    obs, r, term, trunc, info = stack.step(CalvinPoseEnv.hold_action(+1))
    assert info["gripper_open"] < 50 and info["metrics"]["steps_taken"] == 2


def test_the_macro_budget_is_per_subtask(stack):
    """A sub-task that spends its moves without closing ends the chain — the
    evaluator returns at the first failure."""
    obs, info = stack.reset(options={"episode": 0})
    for _ in range(C.MACRO_STEPS_PER_SUBTASK):
        obs, r, term, trunc, info = stack.step(CalvinPoseEnv.hold_action(-1.0))
        if trunc or term:
            break
    assert trunc and not term
    assert info["metrics"]["steps_taken"] <= C.MACRO_STEPS_PER_SUBTASK
    with pytest.raises(RuntimeError):
        stack.unwrapped.step(CalvinPoseEnv.hold_action(-1.0))


def test_env_checker(stack):
    from gymnasium.utils.env_checker import check_env

    check_env(stack.unwrapped, skip_render_check=True)


def test_upstream_protocol_is_calvins_own_tick():
    pytest.importorskip("calvin_env")
    if not (os.environ.get("EMBODIEDSCORE_RUN_CALVIN_TESTS") and _data_root()):
        pytest.skip("EMBODIEDSCORE_RUN_CALVIN_TESTS / EMBODIEDSCORE_DATA_ROOT not set")
    env = es.make(LINE + "-upstream", "mini")
    try:
        obs, info = env.reset(options={"episode": 0})
        body = bodies.CALVIN
        assert obs["rgb"].shape == (body.rgb.height, body.rgb.width, 3)
        assert obs["wrist"].shape == (body.wrist.height, body.wrist.width, 3)
        assert isinstance(env.unwrapped, CalvinEnv) and not isinstance(env.unwrapped, CalvinPoseEnv)
        assert env.unwrapped.action_space.shape == (ACTION_DIM,)
        assert info["subtask_tick_cap"] == C.EP_LEN and info["subtask_move_cap"] is None
        for n in range(3):
            obs, r, term, trunc, info = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
            assert not trunc and info["metrics"]["steps_taken"] == n + 1
            assert info["ticks"] == n + 1
    finally:
        env.close()
