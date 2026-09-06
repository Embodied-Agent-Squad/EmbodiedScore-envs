# Parity with the habitat-lab 0.1.7 / VLN-CE evaluator

`scripts/parity_replay.py` replays board runs that were scored under habitat-sim 0.1.7 +
habitat-lab 0.1.7 + VLN-CE, through the Gymnasium stack (`make_vlnce`) on habitat-sim 0.3.3
(EmbodiedScore-habitat), and compares per-step agent positions, the final distance-to-goal
and all seven metrics.

## 2026-09-06 — R2R-CE rand100 + RxR-CE rand100 (gpt-6 bare board runs)

Stack: `ac-es` (Python 3.11.16), habitat-sim 0.3.3 built with `build.sh`, gymnasium 1.3.0,
fastdtw 0.3.4 **Cython build**, embodiedscore-envs `931da00`+.

| run | episodes | positions | distance_to_goal | metrics (7) | result |
|---|---|---|---|---|---|
| `std_r2r_codex_gpt-6_default_bare` | 100 | max Δ 0.000000 m at every step | max Δ 0.000000 m | max Δ 1e-6 | **100 / 100 ok** |
| `std_rxr_codex_gpt-6_default_bare` | 100 | max Δ 0.000000 m | max Δ 1e-6 m | max Δ 1e-6 | **99 / 99 ok** + 1 truth hole |

- The truth hole is RxR rand100 index 10 (episode 1575): the goal is unreachable on the
  shipped navmesh, so habitat 0.1.7 recorded `distance_to_goal = inf` and no metrics. This
  stack returns `inf` and `info["metrics_valid"] = False` for the same episode.
- 1e-6 residuals are float32→JSON representation of the truth files, not computation.
- With fastdtw's pure-Python fallback instead of the Cython build, one RxR episode (index
  30, 294 steps) differs in nDTW by 1.2e-4; everything else is unchanged. The pure-Python and
  Cython fastdtw are not bit-identical; the 0.1.7 boards used the Cython build, so this
  package requires it for exact parity (see README).

Tolerances used: positions 1e-4 m, distance_to_goal 1e-3 m, metrics 1e-4.

Reproduce (truth files come from the AgentCanvas workspace's `topdown_truth` pass):

```
EMBODIEDSCORE_DATA_ROOT=<.../data/datasets> EMBODIEDSCORE_SCENE_ROOT=<.../scene_datasets> \
python scripts/parity_replay.py --run <run_dir> --eps 0-99
```
