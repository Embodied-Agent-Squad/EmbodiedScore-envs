# EmbodiedScore-envs

Every benchmark of the EmbodiedScore workspace behind the Gymnasium 1.3 API:
the habitat lines on one simulator — a frozen
[habitat-sim 0.3.3](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat),
no habitat-lab — with the legacy stacks' numerics reproduced to the last digit
(evidence archived on the `archive/reproduction` branch),
[VLNverse](https://arxiv.org/abs/2512.19021) on Isaac Sim 5.1 through an
out-of-process render worker, and the manipulation lines — LIBERO and
RoboCasa / RoboCasa365 — on robosuite / MuJoCo in process.

```python
import embodiedscore_envs as es

env = es.make("objectnav-hm3d-v1", "val")        # gym.make + TimeLimit + metrics + depth post-processing
obs, info = env.reset(options={"episode": 0})   # or {"episode_id": ...}; no options = a random episode
print(info["episode"]["goal"]["category"])       # what the episode asks for lives in info, never in obs
obs, reward, terminated, truncated, info = env.step(es.Act.FORWARD)
obs, reward, terminated, truncated, info = env.step(es.Act.STOP)
print(info["metrics"])                           # {"distance_to_goal": ..., "success": ..., "spl": ..., "soft_spl": ...}
env.close()
```

| line (`make` name) | splits | actions (standard) | task semantics (both variants) | original evaluator |
|---|---|---|---|---|
| `vlnce-r2r` · `vlnce-rxr` | train / val_seen / val_unseen | 0–5 | the seven VLN-CE metrics, success 3.0 m | VLN-CE on habitat-lab 0.1.7 |
| `ivlnce` | train / val_seen / val_unseen | 0–5 | VLN-CE metrics + tour t-nDTW (`info["tour"]`) | IVLN-CE on habitat-lab 0.1.7 |
| `objectnav-hm3d-v1` · `-hm3d-v2` · `-mp3d-v1` | train / val / val_mini | 0–5 | distance_to_goal / success 0.1 m / spl / soft_spl | habitat-lab 0.2.4 ObjectNav-v1 |
| `ovon` | val_seen / val_seen_synonyms / val_unseen | 0–5 | same keys; success 0.25 m, goal widened by children categories (OVON's measure) | OVON on habitat-lab 0.2.4 |
| `goat` | val_seen / val_seen_synonyms / val_unseen | 0–6 (+ SUBTASK_STOP) | GOAT-Bench partial / composite metrics, 0.25 m | goat-bench on habitat-lab 0.2.4 |
| `hmeqa` · `mthm3d` | val / mip100 | 0–5 | path_length / steps_taken; `num_step` budget in info; answer scored outside | explore-eqa / MemoryEQA on habitat-sim 0.3.3 |
| `express` | val / train / all / mip100 | 0–5 | distance_to_goal (d_T) / path_length / steps_taken | EXPRESS-Bench on habitat-sim 0.3.3 |
| `vlnverse-fine` · `vlnverse-coarse` | train / val / val_unseen / test / challenge | 0–3 | VLNverse's own formulas (§ Metrics), success 3.0 m, budget 500 | VLNverse evaluator (InternUtopia) — formulas only; see § Fidelity |
| `libero-spatial` · `-object` · `-goal` · `-10` · `-90` | all / mini | — (pose / OSC) | success / steps_taken / ticks — the BDDL predicate check | LIBERO on robosuite 1.4.1 |
| `libero-pro-spatial` · `-object` · `-goal` · `-10` | all / mini | — (pose / OSC) | same keys; the suite's ten tasks perturbed four ways | LIBERO-PRO (arXiv:2510.03827) |
| `libero-plus-camera` · `-noise` · `-robot` · `-language` · `-layout` · `-light` · `-background` | all / mini | — (pose / OSC) | same keys; one perturbation kind over all four suites | LIBERO-Plus (arXiv:2510.13626) |
| `robotwin-clean` · `robotwin-randomized` | all / mini | — (Box) | success / steps_taken / ticks, the task's own `check_success` | RoboTwin 2.0 on SAPIEN 3 |
| `robocasa365-atomic-seen` · `-composite-seen` · `-composite-unseen` | all / mini | — (pose + base / 12-D) | success / steps_taken / ticks, RoboCasa's own `_check_success` | RoboCasa365 on robosuite 1.5 |
| `robocasa-pnp` · `-doors` · `-drawers` · `-levers` · `-knobs` · `-insertion` · `-buttons` · `-navigate` | all / mini | — (pose + base / 12-D) | same | RoboCasa v0.2 on robosuite 1.5 |

The last column is provenance only — which evaluator each line's numbers were
reproduced against (evidence on the `archive/reproduction` branch). The habitat
lines run on habitat-sim 0.3.3; the VLNverse lines on Isaac Sim 5.1; the LIBERO
lines on robosuite / MuJoCo; the RoboTwin lines on SAPIEN 3 (`Benchmark.engine`).

Every line has two variants. `make(name, split)` is the **standard** variant:
EmbodiedScore's shared body (`presets.bodies.STANDARD` — 0.25 m / 15°, tilt 30°
clamped to ±60°, 1.5 m / 0.1 m agent, sliding on, RGB 512² + depth 256² at hfov
90 and 1.25 m, depth 0–10 m normalised, the scene's navmesh file). `make(name,
split, variant="upstream")` (== `make(f"{name}-upstream", split)`) is the line's
own evaluator: VLN-CE's 224² rig, RxR-CE's LoCoBot rig with LOOK, habitat-lab's
LoCoBot for ObjectNav, the Stretch rig (RGB only, navmesh climb 0.1 / cell 0.05)
for OVON and GOAT, and the free-pose teleport protocols (`Box[x, z, yaw]`) of
explore-eqa / EXPRESS for the EQA lines. Task semantics do not change between
variants; only the body, action table, depth post-processing and — for the EQA
and VLNverse lines — the action interface do.

The VLNverse lines cannot run a habitat `Body` (the agent is a camera on an
occupancy grid: no navmesh, no tilt). Their standard variant is the STANDARD
numbers on an `IsaacBody` (`presets.bodies.VLNVERSE_STANDARD`: 0.25 m / 15°,
camera 1.25 m, RGB 512² + depth 256², depth 0–10 m normalised, discrete 0–3);
`-upstream` is the zero-shot line's rig (RGB-D 1024² at 1.2 m, depth in metres)
behind that line's macro action, `Box[angle_rad, distance_m, elevation_deg]`.

Action ids are global: `0 STOP · 1 FORWARD · 2 LEFT · 3 RIGHT · 4 LOOK_UP · 5 LOOK_DOWN · 6 SUBTASK_STOP`;
each declaration uses a prefix. `obs = {"rgb": uint8 HxWx3[, "depth": float32 HxWx1]}`
with the variant's rig; `info` carries the facts (position, rotation,
heading, pitch, collided, distance_to_goal, stop_called, goal_index) plus
`episode` / `goal` on reset and `metrics` from the metric wrapper. Poses are
in the engine's frame — the one the split files use: habitat lines y up with
`(x, y, z, w)` quaternions, VLNverse z up with `(w, x, y, z)`.

## Layout = call chain = semantic hierarchy

```
embodiedscore_envs/
├── __init__.py                 make(name, split, **kw); the gym ids
└── benchmarks/                 one file per line — declarations (standard + upstream): engine, body, action prefix, episodes(), metrics, budget
    ├── vlnce.py  ivlnce.py  objectnav.py (+ ovon)  goat.py  hmeqa.py (+ mthm3d)  express.py  vlnverse.py
    ├── libero.py  libero_pro.py  libero_plus.py     the three manipulation benchmarks on the one LIBERO engine
    │   robotwin.py  robocasa.py  behavior.py
    ├── presets/                bodies (STANDARD + the upstream rigs, habitat and Isaac) · action tables · depth post-processing — pure data
    └── env/                    what every benchmark is built on
        ├── schema.py           Episode · Goal types (Point / Object / Image / Text / Sequence / Question / Manip) · Act · Benchmark
        ├── habitat_env.py      HabitatEnv (Discrete) · HabitatPoseEnv (Box teleport)
        ├── isaac_env.py        IsaacEnv (Discrete) · IsaacPolarEnv (Box rotate-then-advance)
        ├── libero_env.py       LiberoEnv (Box: per-tick OSC delta) · LiberoPoseEnv (Box: absolute end-effector target, closed loop)
        ├── robotwin_env.py     RobotwinEnv (Box: RoboTwin's own joint / end-effector action) · RobotwinPoseEnv (Box: an absolute end-effector target PER ARM, closed loop)
        ├── robocasa_env.py     RobocasaEnv (Box: RoboCasa's per-tick 12-D action) · RobocasaPoseEnv (Box: base move | absolute end-effector target | gripper hold)
        ├── behavior_env.py     BehaviorEnv (Box: the BEHAVIOR Challenge's per-tick 23-D action) · BehaviorPoseEnv (Box: base move | an absolute end-effector target PER ARM | gripper hold)
        ├── metrics.py          NavMetrics (one formula set) · SequenceNavMetrics (GOAT) · VLNVerseMetrics · ManipMetrics (LIBERO, RoboTwin, RoboCasa) · BehaviorMetrics (the BEHAVIOR Challenge's q_score + efficiency terms)
        ├── wrappers.py         DepthClip · DynamicTimeLimit
        └── sim/                one facade per engine; the engines never import each other
            ├── habitat/        the only place that imports habitat_sim (lazily): Body · SceneRef · SimWorld
            ├── isaac/          IsaacBody · IsaacSceneRef · IsaacWorld; worker.py runs under Isaac's python, backend.py drives it
            ├── libero/         LiberoBody · LiberoSceneRef · LiberoWorld — the only place that imports libero / robosuite (lazily)
            ├── robotwin/       RobotwinBody · RobotwinSceneRef · RobotwinWorld — the only place that imports RoboTwin / sapien (lazily)
            ├── robocasa/       RobocasaBody · RobocasaSceneRef · RobocasaWorld — the only place that imports robocasa / robosuite (lazily)
            └── behavior/       BehaviorBody · BehaviorSceneRef · BehaviorWorld — the only place that imports omnigibson / Isaac Sim 4.5 (lazily; Isaac boots once per process)
```

Imports point strictly downward; benchmark files never import each other
(shared pieces live in `presets/` and `env/`); metric wrappers read only `info`.
`import embodiedscore_envs` needs no simulator — a line asks for its engine
when it is built.

## Data

`EMBODIEDSCORE_ROBOTWIN_ROOT` points at the RoboTwin checkout (the RoboTwin
lines read their tasks, configs, language and meshes from inside it —
INSTALL-robotwin.md). `EMBODIEDSCORE_DATA_ROOT` holds one directory per corpus (`vlnce/`, `ivlnce/`,
`objectnav/`, `ovon/`, `goat_bench/`, `hmeqa/`, `mt_hm3d/`, `express_bench/`,
`vlnverse/`) and `EMBODIEDSCORE_SCENE_ROOT` the scenes (`mp3d/`, `hm3d/`,
`hm3d_v0.2/`, `hm3dsem/`, `vlnverse/`); the exact files each loader reads are
in its module docstring. Both can be overridden per call:
`es.make(name, split, data_root=..., scene_root=...)`. The VLNverse release is
fetched by `scripts/download_vlnverse_data.py` (INSTALL-isaac.md § 5). The
LIBERO lines read nothing from either root: the BDDL files and init states ship
inside the `libero` package (INSTALL-libero.md); the RoboTwin lines read
neither root. Nor do the RoboCasa lines: a
kitchen is generated from a layout id and a style id, and the meshes and
textures it draws from are downloaded into the `robocasa` checkout
(INSTALL-robocasa.md). The BEHAVIOR line reads neither root either: OmniGibson
resolves its own assets by `OMNIGIBSON_DATA_PATH`, and the loader is given the
same root (`es.make("behavior-1k", split, behavior_root=...)`, else that
variable) — INSTALL-behavior.md § 4.

## The LIBERO lines (manipulation)

`libero-spatial` / `-object` / `-goal` / `-10` / `-90`: a Franka Panda under
robosuite's OSC_POSE controller, one BDDL task per episode family, 50 init states
per task (`benchmarks/libero.py`). Splits `all` and `mini` (init states 0–9).
Success is the simulator's own predicate check; `ManipMetrics` reports `success`,
`steps_taken`, `ticks` and nothing else, as LIBERO does.

| | `<line>` (standard) | `<line>-upstream` |
|---|---|---|
| body | `LiberoPoseEnv`: one step = one absolute end-effector target `[x, y, z, ax, ay, az, gripper]` (world frame, metres, axis-angle), driven in a closed loop of bounded OSC deltas (≤ 2 cm / 0.05 rad of goal shift per tick, 1.5 cm / 0.1 rad tolerance, ≤ 200 ticks or a stall); a target at `inf` is a 60-tick gripper hold | `LiberoEnv`: one step = one 20 Hz tick, the action is the OSC_POSE input in [−1, 1] (unit = 5 cm / 0.5 rad), as LIBERO's evaluators drive it |
| rig | 256² agentview + wrist (`bodies.LIBERO_STANDARD`) | 128² (`bodies.LIBERO`, LIBERO's `img_h`/`img_w`) |
| budget | 100 macro steps (gym TimeLimit) + a tick guard of 10× the upstream cap | OpenVLA's per-suite tick cap: 220 / 280 / 300 / 520 / 400 |
| termination | the agent's own stop or the budget; `success` latches the first tick the goal held | LIBERO's: done on success |

The budget unit differs between the variants (macro moves vs ticks) — the one
place the "task semantics do not change between variants" rule is bent, for the
same reason the VLNverse lines carry a polar macro action: a language agent
cannot emit 20 Hz deltas. Frames are turned upright (robosuite renders them
upside down; every LIBERO consumer flips them). Object poses are in `info`
(privileged) and never in the observation.

## The LIBERO-PRO and LIBERO-Plus lines (robustness)

Two 2025 benchmarks ask the same question of the same simulator — does a policy
that scores >90% on LIBERO understand the task, or has it memorised it? Both
perturb LIBERO's four evaluation suites and keep everything else: the Panda, the
OSC_POSE controller, the BDDL success predicate. They are therefore lines on the
`libero` engine, not engines of their own — same `LiberoWorld`, same
`LiberoEnv` / `LiberoPoseEnv`, same `ManipMetrics`, same two variants. Each
ships its own `libero` package, so each needs its own interpreter
(INSTALL-libero.md § LIBERO-PRO, § LIBERO-Plus).

| | `libero-pro-*` | `libero-plus-*` |
|---|---|---|
| paper / repo | [arXiv:2510.03827](https://arxiv.org/abs/2510.03827) · Zxy-MLlab/LIBERO-PRO `eafdb80` (a fork of LIBERO) | [arXiv:2510.13626](https://arxiv.org/abs/2510.13626) · sylvestf/LIBERO-plus `4976dc3` (a drop-in replacement) |
| the line is | a base suite: `-spatial` `-object` `-goal` `-10` | a perturbation kind: `-camera` `-noise` `-robot` `-language` `-layout` `-light` `-background` |
| perturbations | object (target and receptacle become other categories) · position (two objects trade regions) · semantic (the instruction paraphrased) · task (the goal predicate replaced) | camera viewpoint · sensor noise · robot initial state · language · object layout · light · background texture |
| tasks per line | 10 base tasks × 4 dimensions = 40 | 1599 / 1601 / 1550 / 1537 / 1525 / 1142 / 1076 = 10 030 |
| where it lives | perturbed BDDL + init files (the HuggingFace release `zhouxueyang/LIBERO-Pro`, unpacked into the fork) | BDDL files for background / light / layout; camera, robot, language and noise are encoded in the task NAME and decoded by the fork's `env_wrapper.py` |
| trials per task | 50 init states, as LIBERO | 1 (`num_trials_per_task = 1`) |
| `all` / `mini` | every init state (2000) / states 0–9 (400) | every task, init state 0 / the first 10 tasks of each base suite (40) |
| tick cap | the base suite's, per line | the base suite's, per episode (`info["max_ticks"]`) — a line spans all four |

`episode_id` names the perturbation: `libero_spatial_swap/3/7` on LIBERO-PRO
(the fork's own suite folder), `libero_spatial/608/0` on LIBERO-Plus (the row of
its `task_classification.json`). Neither needed a hook in `LiberoBody` or
`LiberoWorld`: each fork applies its perturbations inside the same
`OffScreenRenderEnv` the World already builds.

One deviation each, both in the instruction and both for the same reason — the
LIBERO registry derives a task's language from its *filename*
(`grab_language_from_filename`), which the perturbation does not update. On
LIBERO-PRO the loader reads the BDDL's own `(:language ...)` field, where
`perturbation.py` writes the paraphrase (the registry would report the
unperturbed sentence and erase the `semantic` dimension). On LIBERO-Plus it
strips the encoded suffix, which the registry leaves in the sentence
("… place it on the plate view 0 0 100 2 352 initstate 0") — not an instruction,
and a leak of the perturbation to the agent; the language line keeps the fork's
own paraphrase, which it reads from a real BDDL. Both are in the module
docstrings.

## The RoboTwin lines (bimanual manipulation)

`robotwin-clean` / `robotwin-randomized`: [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)
(ICML 2026) on SAPIEN 3 — a dual-arm tabletop robot (`aloha-agilex`: two 6-DoF
arms, one parallel gripper each) over the benchmark's 50 tasks, each with its own
`check_success` predicate written in the task's module. The two lines are
RoboTwin's two shipped task configs, which is the axis its own evaluation is
parameterised on (`--task-config demo_clean | demo_randomized`) and the axis
RoboTwin 2.0 reports separately along; the 50 tasks are episode families inside a
line, because RoboTwin's own eval config lists all 50 flat and its leaderboard is
a per-task success rate averaged over them.

Episodes are task × scene seed. RoboTwin's evaluator walks upward from seed
100000 and keeps the first 100 seeds per task whose scene settles; here each
episode owns a 50-seed window of that same range and takes the first settled
seed in it, so an episode index means one scene on every run
(`benchmarks/robotwin.py`). Splits `all` (100 windows × 50 tasks = 5000) and
`mini` (2 windows × 50 tasks = 100). `ManipMetrics` reports `success`,
`steps_taken` and `ticks`, as RoboTwin reports success rate and nothing else.

| | `<line>` (standard) | `<line>-upstream` |
|---|---|---|
| body | `RobotwinPoseEnv`: one step = an absolute end-effector target PER ARM, `[x, y, z, ax, ay, az, gripper] × 2` (world frame, metres, axis-angle); an arm whose position is `inf` HOLDS its pose and only its gripper command applies, which is how a one-armed move is written | `RobotwinEnv`: one step = one `take_action` call with RoboTwin's own flat joint vector `[left arm, left gripper, right arm, right gripper]`, as its evaluator drives it |
| rig | head + one wrist camera per arm at `Large_D435`, 640×480 (`bodies.ROBOTWIN_STANDARD`) | the same rig at `D435`, 320×240 — what both task configs name (`bodies.ROBOTWIN`) |
| budget | 100 macro steps (gym TimeLimit) + the task's own action cap underneath | the task's own cap, 400–1700 by task (`_eval_step_limit.yml`) |
| termination | the agent's own stop or the budget; `success` latches the first action the goal held | RoboTwin's: done on success or the cap |

The budget unit differs between the variants, as on the LIBERO lines and for the
same reason. Unlike LIBERO, one macro move costs about ONE upstream tick rather
than a hundred: a RoboTwin action is already a planned motion, not a 20 Hz
delta. Actor poses are in `info` (privileged) and never in the observation.

## The RoboCasa lines (kitchen manipulation on a mobile manipulator)

`robocasa365-*` (RoboCasa365, ICLR 2026, [arXiv:2603.04356](https://arxiv.org/abs/2603.04356);
repo `main`, version 1.0.1) and `robocasa-*` (RoboCasa, RSS 2024,
[arXiv:2406.02523](https://arxiv.org/abs/2406.02523); repo tag `v0.2`) are one
engine over two releases of the same package. The robot is a Franka Panda on an
Omron mobile base (`PandaOmron`) under robosuite's `HYBRID_MOBILE_BASE`
controller: the arm on `OSC_POSE` at 20 Hz (5 cm / 0.5 rad per unit **in the
arm's base frame**), the torso on `JOINT_POSITION`, the base on
`JOINT_VELOCITY`. Success is the task class's own `_check_success`;
`ManipMetrics` reports `success`, `steps_taken` and `ticks`, as on the LIBERO
lines.

Lines are the groups each release itself scores — RoboCasa365's three target
task sets (`TASK_SET_REGISTRY`, 18 + 16 + 16 = the leaderboard's 50), RoboCasa
v0.2's eight foundational skills over its 25 atomic tasks. Splits `all` and
`mini`.

An episode is one evaluation scenario: `(task, layout_id, style_id, seed)`.
RoboCasa365's `target` split is 10 fixed kitchens `(1, 1) … (10, 10)` with the
target object instances, RoboCasa v0.2's is 5 — `(1, 1), (2, 2), (4, 4), (6, 9),
(7, 10)` — with object split `B`; both prescribe 50 rollouts per task. Neither
says *which* scenario is which, so this port deals the 50 round-robin over the
line's scenes and uses the scenario index as the env seed:
`episode_id = <task>/<layout>-<style>/<seed>`. Every reset regenerates the
kitchen (fixtures, objects, placements, robot spawn), which is why a reset costs
tens of seconds.

| | `<line>` (standard) | `<line>-upstream` |
|---|---|---|
| body | `RobocasaPoseEnv`: one step is one **base move** `[dx, dy, dyaw]` in the base's own frame, one **absolute end-effector target** `[x, y, z, ax, ay, az, gripper]` driven in a closed loop of bounded OSC deltas (≤ 2 cm / 0.05 rad per tick, 1 cm / 0.1 rad tolerance, ≤ 200 ticks; the base loop parks to 5 cm / 0.05 rad in ≤ 300), or a **gripper hold** — a target at `inf` with no base move, 60 ticks, as on the LIBERO lines | `RobocasaEnv`: one step = one 20 Hz tick, the action is RoboCasa's own 12-D gym action (eef delta 6, gripper flag, base 3, torso, base-mode flag), as `robocasa/wrappers/gym_wrapper.py` composes it |
| rig | `robot0_agentview_center` 256² + wrist (`bodies.ROBOCASA_STANDARD`) | RoboCasa's three cameras at 128²: `agentview_left` as `rgb`, `agentview_right` as `aux`, `eye_in_hand` as `wrist` (`bodies.ROBOCASA`, `create_env`'s defaults) |
| budget | 100 macro steps (gym TimeLimit) + a tick guard of 10× the task's horizon | the task's own horizon (`dataset_registry.py`; per task, 300–7200 on RoboCasa365) |
| termination | the agent's own stop or the budget; `success` latches the first tick the check held | RoboCasa's: done on success |

The budget unit differs between the variants for the same reason as on the
LIBERO lines. Frames are turned upright (RoboCasa's own gym wrapper flips the
renderer's rows too). Object poses, the fixture list and the episode's language
are in `info` (`info["language"]` is the instruction RoboCasa words for the
objects it sampled — only known after a reset, so it also replaces the
declaration's placeholder in `info["episode"]["instruction"]`).

## The BEHAVIOR-1K line (long-horizon household activities)

`behavior-1k`: the 2025 BEHAVIOR Challenge (NeurIPS 2025) on BEHAVIOR-1K
([arXiv:2403.09227](https://arxiv.org/abs/2403.09227)), pinned at
`StanfordVL/BEHAVIOR-1K` tag `v3.7.2` (commit `88454bd0`) — OmniGibson 3.7.2 on
NVIDIA Isaac Sim 4.5.0, a different Isaac from the VLNverse lines' 5.1, so a
different conda environment (INSTALL-behavior.md). The robot the challenge
fixes is a **Galaxea R1 Pro** (`eval.py` L128 asserts it): a holonomic wheeled
base, a four-joint torso, two 7-DOF arms with a parallel gripper each, 28 DOF,
control at 30 Hz over 120 Hz physics, three RGB cameras (a ZED head camera and
a RealSense in each wrist).

One line, because the leaderboard is one number:
`learning/utils/score_utils.py` averages the per-task score over its 10
instances and then over all 50 tasks, and computes nothing per scene or per
group. The 50 activities span three scene models — `house_single_floor` (23),
`house_double_floor_lower` (22), `house_double_floor_upper` (5).

An episode is one **task instance**: an activity plus one of its sampled
initial states. For each task the port takes the 10 public-test instances the
release names in `metadata/test_instances.csv`, which is exactly an official
submission's 500 rollouts. Splits `all` (500) and `mini` (the first instance of
each task, 50 — one instance per activity rather than a prefix of the task
list, because a scene load costs minutes). `episode_id = <task>/<instance_id>`.

Success is BEHAVIOR's own: every goal predicate of some grounding of the
activity's BDDL goal satisfied. `BehaviorMetrics` reports the challenge's own
set — `q_score` (its ranking metric: 1.0 on success, else the largest fraction,
over the goal's groundings, of predicates that were false at the start and are
true now — `omnigibson/metrics/task_metric.py`), `success`, and the efficiency
terms it normalises by the mean of that task's 200 human demonstrations
(simulated time, and base / left-hand / right-hand travel —
`omnigibson/metrics/agent_metric.py`).

| | `behavior-1k` (standard) | `behavior-1k-upstream` |
|---|---|---|
| body | `BehaviorPoseEnv`: one step is one **base move** `[dx, dy, dyaw]` in the base's own frame (a proportional loop on the holonomic velocity controller, parking to 5 cm / 0.05 rad in ≤ 300 ticks), an **absolute end-effector target per arm** `[x, y, z, ax, ay, az, gripper]` (world frame, metres, axis-angle; an arm at `inf` holds, as on the RoboTwin lines) driven for ≤ 120 ticks to 2 cm / 0.15 rad, or a **gripper hold** — both arms at `inf` with no base move, 30 ticks | `BehaviorEnv`: one step = one 30 Hz tick, the action is the challenge's own 23-D vector `[base 3 | torso 4 | left arm 7 | left gripper 1 | right arm 7 | right gripper 1]` — absolute joint angles for the torso and both arms, a base velocity, a finger target per gripper (`ACTION_QPOS_INDICES["R1Pro"]`, `R1_CONTROLLER_CONFIG`) |
| controllers | the challenge's, with the two arms swapped to `InverseKinematicsController` in `absolute_pose` mode — one of the four substitutions `docs/challenge/evaluation.md` § "Configure Robot Action Space" documents for participants, applied through the hook the evaluator itself uses | the challenge's, verbatim |
| rig | head + both wrists at 256² (`bodies.BEHAVIOR_STANDARD`) | the challenge's `RGBLowResWrapper`: the same three cameras at 224², head aperture 40 mm (`bodies.BEHAVIOR`) |
| budget | 150 macro steps (gym TimeLimit) | — (the tick cap alone) |
| tick cap | the task's own: twice the mean length of its 200 human demos (`eval.py` L146-153; 4 299 to 52 120 ticks across the 50 tasks) | the same |
| termination | the agent's own stop or the budget; `success` latches the first tick the goal held | the challenge's: done on success |

**Known limitation of the macro protocol: it does not command the torso.** The
R1 Pro's four trunk joints are in the challenge's own action space and they
decide where the arms can reach at all; the macro protocol holds them at the
posture the instance loaded with. Measured 2026-09-10 on ``turning_on_radio``,
the closed loop asymptotes 0.131 m short of the radio's toggle button from a
0.40 m base park (0.031 m from a 0.55 m park) and then improves by under a
millimetre per 250 ticks — the arm's kinematic limit with the torso fixed, not
a tracking failure. Giving the protocol an absolute torso target, or solving
for one inside the closed loop, is the outstanding work. The per-tick variant
commands the torso already: it is the challenge's own 23-D action.

Only the unit the *agent's* budget counts bends between the variants; the
simulator's tick cap is BEHAVIOR's own on both. A zero action vector is not a
no-op on this robot (the torso and both arms take absolute joint angles) —
`world.no_op_action()` is. Isaac Sim boots once per process
(`simulator.py` L386); the world reloads at the cheapest level a change allows
— a new instance replays a JSON state, a new task in the same house is
`env.update_task`, and only a new scene model clears the stage and rebuilds.
Goal predicates, per-predicate goal status, accumulated travel and the poses of
every task-relevant object are in `info`.

## Metrics of the VLNverse lines

`VLNVerseMetrics` implements the VLNverse evaluator's formulas (`VLNPEMetrics`,
billzhao1030/VLNVerse at `d444c04`). They differ from `NavMetrics` — the
definitions the same names carry on every other line:

| key | VLNverse evaluator's formula | `NavMetrics` (VLN-CE / habitat-lab) |
|---|---|---|
| `distance_to_goal` | euclidean xy to the end of the reference path | navmesh geodesic |
| `success` | distance_to_goal < 3.0 m | also requires STOP |
| `oracle_success` | min distance over post-action positions < 3.0 m | start pose included |
| `spl` | success · S / max(path_length, S), S = reference-path polyline length; 0 if path_length = 0 | S = geodesic; equals success if path_length = 0 |
| `ndtw` | mean over agent positions of exp(−d²/18), d = distance to the nearest reference waypoint | exp(−DTW / (\|ref\| · 3)) with a DTW alignment |
| `path_length` | sum of xy displacements | 3-D |
| `steps_taken` | `step()` calls, STOP included | same |

On `test` / `challenge` the ground-truth keys are −1 and `info["metrics_valid"]` is `False`.

## Fidelity

The port reproduces the legacy evaluators (habitat-lab 0.1.7 + VLN-CE,
habitat-lab 0.2.4, raw habitat-sim 0.3.3) to the last digit: recorded runs
replayed through this stack give the same per-step positions, distance-to-goal,
metrics, camera pitch and rendered frames. The replay tooling, oracle
generators and per-episode logs are archived on the `archive/reproduction`
branch (`reproduction/REPRODUCTION.md` there has the numbers).

Two things the reproduction work showed that a re-implementation must copy:
habitat-sim's `MultiGoalShortestPath` prunes with bounds carried over from the
previous query, so `distance_to_goal` depends on *when* it is queried (the
habitat-lab call pattern — only when the agent moved — is reproduced, and
goat-bench's every-step pattern for GOAT); and explore-eqa pitches the whole
agent, so its camera swings about the agent's feet (`Body.pitch_moves_camera`).

The VLNverse lines reproduce the evaluator's *formulas* and data, not its
physics: the original evaluator moved a Unitree H1 in InternUtopia; here the
agent is a camera moved kinematically on the scene's occupancy grid — the
zero-shot line's protocol (billzhao1030/vlnverse_emr_zero_shot), which the
`-upstream` variant carries as-is. Isaac's RTX frames are not bit-identical
between renders of one pose (facts and depth are); the gym ids register
`nondeterministic=True`.

The RoboTwin lines reproduce RoboTwin's physics as is (SAPIEN 3 at its 1/250 s
step, the shipped embodiment, task configs and per-task step caps; the curobo
planner its embodiment config names). Two things differ from upstream and are
declared: the episode-to-seed mapping (per-episode seed windows instead of a
shared queue, so an episode index is reproducible) and the instruction (the
task's `full_description` rather than a per-episode template, which only
RoboTwin's scripted expert can fill in). Its Vulkan ray-traced frames are not
bit-identical between renders of one state, so its gym ids register
`nondeterministic=True` too.

The RoboCasa lines reproduce RoboCasa's physics as is (robosuite 1.5, the
release's controller config and task classes). What they do *not* inherit is
the choice of scenario: RoboCasa draws each of its 50 evaluation rollouts from
the split's scenes at random, and this port deals them round-robin instead, so
an episode id names a reproducible kitchen (§ The RoboCasa lines).

The LIBERO lines reproduce LIBERO's physics as is (robosuite 1.4.1, the
controller config and the init states are the release's; MuJoCo is deterministic
given the state — facts and proprioception repeat exactly). Its offscreen EGL
frames are not bit-identical between renders of one state; the gym ids register
`nondeterministic=True` too.

## Install

1. habitat-sim 0.3.3 from EmbodiedScore-habitat (`./build.sh --env <env> --verify`) — habitat lines.
2. `pip install -e .` in the same environment (adds gymnasium, numpy-quaternion, scipy, fastdtw, msgpack).
   For nDTW to match the habitat-lab 0.1.7 boards to the last digit, fastdtw must be its
   Cython build (the pip wheel silently falls back to pure Python, which differs by ~1e-4):
   `pip install cython && pip install --no-cache-dir --no-binary fastdtw --no-build-isolation fastdtw`.
   `NavMetrics` warns once if the fallback is in use.
3. Isaac Sim 5.1 — VLNverse lines only: [INSTALL-isaac.md](INSTALL-isaac.md) (pip, workstation
   bundle, or the container launcher in `scripts/`); point `EMBODIEDSCORE_ISAAC_PYTHON` at its python.
4. LIBERO (robosuite 1.4.1 + MuJoCo) — the `libero-*` lines only, in their own env:
   [INSTALL-libero.md](INSTALL-libero.md). LIBERO-PRO and LIBERO-Plus each ship their own
   `libero` package, so each gets an env of its own (§ LIBERO-PRO, § LIBERO-Plus there):
   `ac-libero` · `ac-libero-pro` · `ac-libero-plus`.
5. RoboTwin 2.0 (SAPIEN 3 + curobo) — the `robotwin-*` lines only, in their own env:
   [INSTALL-robotwin.md](INSTALL-robotwin.md).
6. RoboCasa (robosuite 1.5 + MuJoCo) — the `robocasa*-*` lines only, in *two* more envs
   (the two releases pin incompatible numpy / mujoco / python and share a distribution
   name): [INSTALL-robocasa.md](INSTALL-robocasa.md).
7. `pytest tests` with the data roots set (a GPU and the datasets are needed); the Isaac
   contracts additionally opt in with `EMBODIEDSCORE_RUN_ISAAC_TESTS=1`, the LIBERO ones with
   `EMBODIEDSCORE_RUN_LIBERO_TESTS=1`, `EMBODIEDSCORE_RUN_LIBERO_PRO_TESTS=1` and
   `EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS=1` — each in the env that has its own `libero` — the
   RoboTwin ones with `EMBODIEDSCORE_RUN_ROBOTWIN_TESTS=1`, the RoboCasa ones with
   `EMBODIEDSCORE_RUN_ROBOCASA_TESTS=1`.

Python 3.10–3.12, Linux, NVIDIA driver for headless EGL (and Vulkan for the
RoboTwin lines).
