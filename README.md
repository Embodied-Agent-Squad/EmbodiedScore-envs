# EmbodiedScore-envs

Every habitat benchmark of the EmbodiedScore workspace on one simulator —
a frozen [habitat-sim 0.3.3](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat),
no habitat-lab — behind the Gymnasium 1.3 API, with the legacy stacks'
numerics reproduced to the last digit (evidence archived on the
`archive/reproduction` branch).

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

The last column is provenance only — which evaluator each line's numbers were
reproduced against (evidence on the `archive/reproduction` branch). Everything
runs on habitat-sim 0.3.3.

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
lines — the action interface do.

Action ids are global: `0 STOP · 1 FORWARD · 2 LEFT · 3 RIGHT · 4 LOOK_UP · 5 LOOK_DOWN · 6 SUBTASK_STOP`;
each declaration uses a prefix. `obs = {"rgb": uint8 HxWx3[, "depth": float32 HxWx1]}`
with the variant's rig; `info` carries the facts (position, rotation,
heading, pitch, collided, distance_to_goal, stop_called, goal_index) plus
`episode` / `goal` on reset and `metrics` from the metric wrapper.

## Layout = call chain = semantic hierarchy

```
embodiedscore_envs/
├── __init__.py                 make(name, split, **kw); the gym ids
└── benchmarks/                 one file per line — declarations (standard + upstream): body, action prefix, episodes(), metrics, budget
    ├── vlnce.py  ivlnce.py  objectnav.py (+ ovon)  goat.py  hmeqa.py (+ mthm3d)  express.py
    ├── presets/                bodies (STANDARD + the upstream rigs) · action tables · depth post-processing — pure data
    └── env/                    what every benchmark is built on
        ├── schema.py           Episode · Goal types (Point / Object / Image / Text / Sequence / Question) · Act · Benchmark
        ├── env.py              HabitatEnv (Discrete) · HabitatPoseEnv (Box teleport)
        ├── metrics.py          NavMetrics (one formula set) · SequenceNavMetrics (GOAT)
        ├── wrappers.py         DepthClip · DynamicTimeLimit
        └── sim/                the only place that imports habitat_sim: Body · SceneRef · SimWorld
```

Imports point strictly downward; benchmark files never import each other
(shared pieces live in `presets/` and `env/`); metric wrappers read only `info`.

## Data

`EMBODIEDSCORE_DATA_ROOT` holds one directory per corpus (`vlnce/`, `ivlnce/`,
`objectnav/`, `ovon/`, `goat_bench/`, `hmeqa/`, `mt_hm3d/`, `express_bench/`)
and `EMBODIEDSCORE_SCENE_ROOT` the scenes (`mp3d/`, `hm3d/`, `hm3d_v0.2/`,
`hm3dsem/`); the exact files each loader reads are in its module docstring.
Both can be overridden per call: `es.make(name, split, data_root=..., scene_root=...)`.

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

## Install

1. habitat-sim 0.3.3 from EmbodiedScore-habitat (`./build.sh --env <env> --verify`).
2. `pip install -e .` in the same environment (adds gymnasium, numpy-quaternion, scipy, fastdtw).
   For nDTW to match the habitat-lab 0.1.7 boards to the last digit, fastdtw must be its
   Cython build (the pip wheel silently falls back to pure Python, which differs by ~1e-4):
   `pip install cython && pip install --no-cache-dir --no-binary fastdtw --no-build-isolation fastdtw`.
   `NavMetrics` warns once if the fallback is in use.
3. `pytest tests` with the data roots set (a GPU and the datasets are needed).

Python 3.10–3.12, Linux, NVIDIA driver for headless EGL.
