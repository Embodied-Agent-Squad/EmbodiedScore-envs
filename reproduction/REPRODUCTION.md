# Reproduction of the legacy stacks

Every benchmark in this package is accepted by replaying recorded action
sequences through the Gymnasium stack on habitat-sim 0.3.3 (EmbodiedScore-habitat,
`ac-es`) and comparing with what the stack that produced the numbers reported:
habitat-lab 0.1.7 + VLN-CE (VLN-CE, IVLN-CE), habitat-lab 0.2.4 (+ vendored
OVON / goat-bench code) (ObjectNav, OVON, GOAT), raw habitat-sim 0.3.3 nodesets
(HM-EQA, MT-HM3D, EXPRESS). `reproduction/run_all.sh` runs everything below;
the per-episode logs are in `reproduction/reports/`.

Tolerances: positions **0** (max Δ reported), float metrics **1e-6**, integer
and boolean metrics exact, camera pitch 1e-3°, rendered frames reported as
mean |Δ| / 255 (never asserted).

## 2026-09-06/07 — the unified package (commit `588ae91`+)

| benchmark | truth | episodes | positions | metrics | result |
|---|---|---|---|---|---|
| `vlnce-r2r` rand100 | 0.1.7 evaluator per-step track + 7 metrics (`topdown_truth`) | 100 | Δ 0 | ≤ 2.4e-7 | **100 / 100** |
| `vlnce-rxr` rand100 | same | 100 | Δ 0 | ≤ 9.5e-7 | **100 / 100** (1 hole: goal unreachable, `inf` both sides) |
| `ivlnce` val_unseen tours 0 / 0 (24 ep) / 7, val_seen tour 3 | legacy `IvlnceEnvManager` oracle (`reproduction/oracle/ivlnce_oracle.py`): episodes, transits, tour path, t-nDTW | 6 + 24 + 6 + 2 = 38 (+ tour paths of 404 / 2143 / 150 / 295 entries) | Δ 0 | Δ 0; transit steps equal (tour 0: 0/57/52/6/100/1); t-nDTW equal to 4 dp | **38 / 38 + all 4 paths identical** |
| `objectnav-hm3d-v1` mip100 | coding-agent transcripts (`std_hm3d_cc_opus-5_default_bare`) + terminal metrics | 100 | — | ≤ 1.8e-8 | **100 / 100** |
| `objectnav-mp3d-v1` mip100 | `std_mp3d_cc_opus-5_default_bare` | 100 | — | ≤ 1e-6 except one episode at 1.9e-6 (build-level float noise in one geodesic) | **99 / 100** |
| `ovon` mip100_seen / _seen_synonyms / _unseen | `std_ovon-{seen,syn,unseen}_cc_opus-5_default_bare` | 3 × 100 | — | Δ 0 except syn ep 23 (see note) | **299 / 300** |
| `goat` val_seen / val_unseen / val_unseen (b) | legacy `GoatEnvManager` oracle (`reproduction/oracle/goat_oracle.py`): per-step track + flattened nested metrics | 36 + 12 + 36 = 84 | Δ 0 | ≤ 1.5e-8 incl. `subtask_success[]`, `spl_by_subtask[]` | **84 / 84** |
| `hmeqa` val | `std_hmeqa500_cc_fable-5_default_bare` actions.log: per-batch camera pitch, `num_step`, frames | 500 | Δ 0 | pitch sequences + budgets exact; frames ≤ 0.02 / 255 | **500 / 500** |
| `hmeqa-pose` | canvas graph log `eval_runs/20260615_183412` (explore-eqa protocol) | 10 | 4e-7 (float32 pose storage) | `num_step`, `floor_height`, truncation step exact | **10 / 10** |
| `mthm3d` mip100 | `std_mthm3d_cc_opus-5_default_bare` | 100 | Δ 0 | exact; frames ≤ 0.01 / 255 | **100 / 100** |
| `express` mip100 | `std_express_cc_opus-5_default_bare`: `path_len`, `d_t`, `gt_geodesic`, `steps_taken`, `num_steps`, frames | 100 | Δ 0 | Δ 0; frames 0.00 / 255 | **100 / 100** |

Notes.

- **`distance_to_goal` depends on the query history.** habitat-sim's
  `MultiGoalShortestPath.find_path` prunes candidate ends with bounds carried
  over from the previous query (`minTheoreticalDist[i] = max(prev − moved, L2)`),
  and Detour's approximate distances do not honour the triangle inequality the
  bound assumes, so an extra or missing query changes the value by centimetres
  (hm3d ep 87 / 96: 8–14 cm before the fix). habitat-lab's stock `DistanceToGoal`
  queries only when the agent moved more than 1e-4 (`np.allclose`), goat-bench's
  every step; `Benchmark.dtg_policy` reproduces each. OVON seen_synonyms ep 23
  is the one episode whose *recorded run* disagrees with a clean replay in the
  legacy stack itself (the run's own history had an extra reset); this package
  matches the clean legacy replay exactly (`reproduction/reports/oracle_ovon_syn_23.json`).
- **explore-eqa pitches the whole agent**, so its camera swings about the
  agent's feet (at −30° it sits 0.75 m ahead and 0.20 m below the nominal
  mount) and moves again on every tilt action. `Body.pitch_moves_camera` plus
  `Body.locomotion="teleport"` (explore-eqa's own float64 yaw / `try_step` /
  `set_state` arithmetic) make HM-EQA / EXPRESS frames pixel-identical and
  EXPRESS `path_len` / `d_t` exact; with habitat's `sim.step` instead they were
  off by 1e-6 (path) and 50–90 / 255 (frames).
- **Scene changes must `close(destroy=False)` before `reconfigure`** — a bare
  reconfigure leaks ~100 MB per scene and a 500-episode run was OOM-killed at
  219; habitat-lab's `should_close_on_new_scene` does the same.
- **Episode identity.** The legacy habitat-lab nodesets index habitat's seed-100
  in-place-shuffled dataset list and ObjectNav / GOAT re-number `episode_id`
  per scene file, so recorded indices are not this package's file order and ids
  repeat across scenes; the replay script selects by `(scene, episode_id)`.
- coding-agent `live_<i>/actions.log` files accumulate every retry of an
  episode; the script keeps the last attempt and skips frame comparison for
  retried episodes.
- VLN-CE nDTW needs fastdtw's **Cython** build (the pure-Python fallback
  differs by ~1e-4 on one RxR episode); `path_length` sums float32 norms as
  habitat does.

Reproduce:

```
EMBODIEDSCORE_DATA_ROOT=<datasets> EMBODIEDSCORE_SCENE_ROOT=<scenes> bash reproduction/run_all.sh
# legacy oracles (run in the legacy envs, see each script's docstring):
PYTHONPATH=agentcanvas/backend ~/miniforge3/envs/ac-objnav/bin/python reproduction/oracle/goat_oracle.py --split val_unseen --eps 0-11 --out reproduction/reports/oracle_goat_val_unseen.json
PYTHONPATH=agentcanvas/backend ~/miniforge3/envs/ac-vlnce/bin/python  reproduction/oracle/ivlnce_oracle.py --split val_unseen --tour 0 --n 6 --out reproduction/reports/oracle_ivlnce_val_unseen_tour0.json
```

## 2026-09-06 — VLN-CE only (commit `10002ea`, pre-unification)

R2R-CE rand100 100 / 100 and RxR-CE rand100 99 / 99 + 1 hole against the 0.1.7
evaluator, positions Δ 0, seven metrics ≤ 1e-6 (`reproduction/reports/2026-09-06_{r2r,rxr}_rand100_gpt6.log`).
