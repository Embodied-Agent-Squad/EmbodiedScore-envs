# EmbodiedScore-envs

Every habitat benchmark of the EmbodiedScore workspace on one simulator —
a frozen [habitat-sim 0.3.3](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat),
no habitat-lab — behind the Gymnasium 1.3 API, with the legacy stacks'
numerics reproduced to the last digit (see `PARITY.md`).

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

| benchmark (`make` name) | splits | actions | metrics |
|---|---|---|---|
| `vlnce-r2r` · `vlnce-rxr` | train / val_seen / val_unseen | 0–3 (0.25 m, 15°) | the seven VLN-CE metrics, success 3.0 m |
| `ivlnce` | train / val_seen / val_unseen | 0–3 | VLN-CE metrics + tour t-nDTW (`info["tour"]`) |
| `objectnav-hm3d-v1` · `-hm3d-v2` · `-mp3d-v1` | train / val / val_mini | 0–5 (0.25 m, 30°, tilt 30°) | distance_to_goal / success 0.1 m / spl / soft_spl |
| `ovon` | val_seen / val_seen_synonyms / val_unseen | 0–5 | same as ObjectNav |
| `goat` | val_seen / val_seen_synonyms / val_unseen | 0–6 (+ SUBTASK_STOP) | GOAT-Bench partial / composite metrics, 0.25 m |
| `hmeqa` · `mthm3d` (+ `-pose`) | val / mip100 | 0–5 (tilt ±60°) / `Box[x, z, yaw]` | path_length / steps_taken; answer scored outside |
| `express` (+ `-pose`) | val / train / all / mip100 | 0–3 / `Box` with navmesh snap | distance_to_goal (d_T) / path_length / steps_taken |

Action ids are global: `0 STOP · 1 FORWARD · 2 LEFT · 3 RIGHT · 4 LOOK_UP · 5 LOOK_DOWN · 6 SUBTASK_STOP`;
each benchmark uses a prefix. `obs = {"rgb": uint8 HxWx3[, "depth": float32 HxWx1]}`
with the benchmark's own rig; `info` carries the facts (position, rotation,
heading, pitch, collided, distance_to_goal, stop_called, goal_index) plus
`episode` / `goal` on reset and `metrics` from the metric wrapper.

## Layout = call chain = semantic hierarchy

```
embodiedscore_envs/
├── __init__.py                 make(name, split, **kw); the gym ids
└── benchmarks/                 one file per benchmark — a declaration: Body, action prefix, episodes(), metrics, budget
    ├── vlnce.py  ivlnce.py  objectnav.py (+ ovon)  goat.py  hmeqa.py (+ mthm3d)  express.py
    └── env/                    what every benchmark is built on
        ├── schema.py           Episode · Goal types (Point / Object / Image / Text / Sequence / Question) · Act · Benchmark
        ├── env.py              HabitatEnv (Discrete) · HabitatPoseEnv (Box teleport)
        ├── metrics.py          NavMetrics (one formula set) · SequenceNavMetrics (GOAT)
        ├── wrappers.py         DepthClip · DynamicTimeLimit
        └── sim/                the only place that imports habitat_sim: Body · SceneRef · SimWorld
```

Imports point strictly downward; benchmark files never import each other
(shared pieces live in `env/`); metric wrappers read only `info`.

## Data

`EMBODIEDSCORE_DATA_ROOT` holds one directory per corpus (`vlnce/`, `ivlnce/`,
`objectnav/`, `ovon/`, `goat_bench/`, `hmeqa/`, `mt_hm3d/`, `express_bench/`)
and `EMBODIEDSCORE_SCENE_ROOT` the scenes (`mp3d/`, `hm3d/`, `hm3d_v0.2/`,
`hm3dsem/`); the exact files each loader reads are in its module docstring.
Both can be overridden per call: `es.make(name, split, data_root=..., scene_root=...)`.

## Fidelity

`scripts/parity_replay.py` replays recorded runs of the legacy stacks
(habitat-lab 0.1.7 + VLN-CE, habitat-lab 0.2.4, raw habitat-sim 0.3.3) and
compares per-step positions, distance-to-goal, every metric, camera pitch and
rendered frames; `scripts/oracle/` produces replay truth in the legacy conda
envs for the benchmarks without recorded runs. `PARITY.md` has the numbers.

Two things the parity work showed that a re-implementation must copy:
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
