# EmbodiedScore-envs

Gymnasium environments for embodied-AI benchmarks, built directly on a frozen
[habitat-sim 0.3.3](https://github.com/Embodied-Agent-Squad/EmbodiedScore-habitat)
— no habitat-lab. First line: **VLN-CE** (R2R-CE and RxR-CE).

```python
import embodiedscore_envs                       # registers the ids
from embodiedscore_envs.vlnce import Act, make_vlnce

env = make_vlnce("r2r", "val_unseen")           # checker / order / TimeLimit(500) / metrics / depth in [0,1]
obs, info = env.reset(options={"episode": 0})   # or {"episode_id": "40"}; no options = random episode
print(info["episode"]["instruction"])
obs, reward, terminated, truncated, info = env.step(Act.FORWARD)
obs, reward, terminated, truncated, info = env.step(Act.STOP)
print(info["metrics"])   # distance_to_goal, success, spl, ndtw, path_length, oracle_success, steps_taken
env.close()
```

`obs = {"rgb": uint8 (512, 512, 3), "depth": float32 (256, 256, 1)}`; actions
`Discrete(4)`: 0 STOP · 1 FORWARD 0.25 m · 2 LEFT 15° · 3 RIGHT 15°. STOP sets
`terminated`; the 500-step budget sets `truncated`.

## Layers

| layer | module | responsibility |
|---|---|---|
| L0 | `embodiedscore_envs.habitat.SimWorld` | one `habitat_sim.Simulator`; load_scene / place / move / pose / geodesic / observe; no episode concept |
| L1 | `embodiedscore_envs.vlnce.VLNCEEnv` | `gym.Env` body: episode list, start placement, one action per step, facts in `info` |
| L2 | `VLNCEMetrics`, `NormalizeDepth`, gymnasium `TimeLimit` | the seven board metrics, depth normalisation, the step budget |
| — | `make_vlnce()` | assembles the standard stack |

The body never counts steps, computes metrics, normalises observations or keeps
a "next episode" cursor; those are wrappers, or the caller's `reset(options=...)`.

## Fidelity

The metrics are transcribed from habitat-lab 0.1.7 and VLN-CE (see
`vlnce/metrics.py`). `scripts/parity_replay.py` replays board runs recorded
under habitat 0.1.7 and compares per-step positions, distance-to-goal and all
seven metrics; the acceptance bar is bit-exact positions and metrics equal to
1e-4. Results are kept in `PARITY.md`.

## Install

1. habitat-sim 0.3.3 from EmbodiedScore-habitat (`./build.sh --env <env> --verify`).
2. `pip install -e .` in the same environment (adds gymnasium, numpy-quaternion, fastdtw).
3. Data: point `EMBODIEDSCORE_DATA_ROOT` at the directory holding
   `R2R_VLNCE_v1-3_preprocessed/` and `RxR_VLNCE_v0/`, and
   `EMBODIEDSCORE_SCENE_ROOT` at the directory holding `mp3d/`.

Python 3.10–3.12, Linux, NVIDIA driver for headless EGL.
