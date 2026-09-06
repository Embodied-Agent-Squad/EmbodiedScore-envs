"""Replay a board run through the Gymnasium stack and compare with the habitat
0.1.7 evaluator truth — the acceptance test of EmbodiedScore-envs' VLN-CE line.

Inputs (a coding-agent std board run directory):
    <run>/summary.json                  config.dataset / config.split, per-episode episode_id
    <run>/episode_<i>.jsonl             transcript; the env_habitat__step tool calls give the actions
    <run>/topdown_truth/ep<i>.json      0.1.7 truth: per-step track, final dtg, seven metrics

For every episode: reset(options={"episode_id": ...}), replay the executed
actions, STOP if the run stopped, then compare per-step positions, final
distance_to_goal and the metrics dict. Prints one line per episode and a
summary; exits 1 if any episode exceeds the tolerances.

Usage:
    EMBODIEDSCORE_DATA_ROOT=... EMBODIEDSCORE_SCENE_ROOT=... \
    python scripts/parity_replay.py --run <run_dir> [--eps 0-99] [--tol-pos 1e-4 --tol-dtg 1e-3 --tol-metric 1e-4]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import embodiedscore_envs  # noqa: F401  (registers the env ids)
from embodiedscore_envs.vlnce import Act, make_vlnce

DATASET_KEY = {"R2R-CE": "r2r", "RxR-CE": "rxr"}
SPLIT_SUFFIXES = ("_en", "_hi", "_te")   # board display split -> loader split


def transcript_actions(run_dir: Path, index: int) -> tuple[list[int], bool]:
    """Movement actions the episode actually executed, and whether it STOPped.
    A step tool call carries a list of actions; its result says how many ran."""
    moves: list[int] = []
    pending: dict[str, list[int]] = {}
    for line in (run_dir / f"episode_{index}.jsonl").read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        kind = d.get("kind")
        if kind == "tool_use" and str(d.get("name", "")).endswith("__step"):
            pending[d.get("id", "")] = [int(a) for a in (d.get("input") or {}).get("actions") or []]
        elif kind == "tool_result" and d.get("tool_use_id") in pending:
            acts = pending.pop(d["tool_use_id"])
            executed = None
            for t in d.get("texts") or []:
                try:
                    v = json.loads(t)
                    if isinstance(v, dict) and "executed" in v:
                        executed = int(v["executed"])
                        break
                except (ValueError, TypeError):
                    pass
            if executed is None:
                continue
            for a in acts[:executed]:
                if a == 0:
                    return moves, True
                if a in (1, 2, 3):
                    moves.append(a)
    return moves, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eps", default="0-99")
    ap.add_argument("--tol-pos", type=float, default=1e-4)
    ap.add_argument("--tol-dtg", type=float, default=1e-3)
    ap.add_argument("--tol-metric", type=float, default=1e-4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run).resolve()
    summary = json.loads((run / "summary.json").read_text())
    cfg = summary["config"]
    dataset = DATASET_KEY[cfg["dataset"]]
    split = cfg["split"]
    for suf in SPLIT_SUFFIXES:
        if dataset == "rxr" and split.endswith(suf):
            split = split[: -len(suf)]
    by_index = {e["index"]: e for e in summary["episodes"]}
    indices: list[int] = []
    for tok in args.eps.split(","):
        a, _, b = tok.partition("-")
        indices.extend(range(int(a), int(b or a) + 1))

    env = make_vlnce(dataset, split, gpu_id=args.gpu)
    worst = {"pos": 0.0, "dtg": 0.0, "metric": 0.0}
    failed: list[int] = []
    holes: list[int] = []
    for i in indices:
        tpath = run / "topdown_truth" / f"ep{i}.json"
        if not tpath.is_file():
            print(f"ep{i:3d}: no truth file — skipped")
            continue
        truth = json.loads(tpath.read_text())
        ep_id = str(truth["episode_id"])
        moves, stopped = transcript_actions(run, i)
        _, info = env.reset(options={"episode_id": ep_id})
        track = [info["position"]]
        for a in moves:
            _, _, term, trunc, info = env.step(a)
            track.append(info["position"])
            if term or trunc:
                break
        if stopped and not info["stop_called"]:
            _, _, _, _, info = env.step(Act.STOP)
        m = info["metrics"]

        tt = np.asarray(truth["track"], dtype=np.float64)
        tr = np.asarray(track, dtype=np.float64)
        n = min(len(tt), len(tr))
        dpos = float(np.linalg.norm(tt[:n] - tr[:n], axis=1).max()) if n else math.inf
        if len(tt) != len(tr):
            dpos = math.inf
        t_dtg = truth.get("dtg")
        t_m = truth.get("metrics") or {}
        if not t_m or t_dtg is None or not math.isfinite(float(t_dtg)):
            holes.append(i)
            print(f"ep{i:3d} id={ep_id:>6}: truth hole (dtg={t_dtg}); ours dtg={m['distance_to_goal']:.4f} valid={info['metrics_valid']}")
            continue
        ddtg = abs(m["distance_to_goal"] - float(t_dtg))
        dmet = max(abs(float(m[k]) - float(t_m[k])) for k in t_m if k in m)
        worst["pos"], worst["dtg"], worst["metric"] = max(worst["pos"], dpos), max(worst["dtg"], ddtg), max(worst["metric"], dmet)
        ok = dpos <= args.tol_pos and ddtg <= args.tol_dtg and dmet <= args.tol_metric
        if not ok:
            failed.append(i)
        print(f"ep{i:3d} id={ep_id:>6}: steps={len(moves):3d} stop={int(stopped)} "
              f"|dpos|max={dpos:.6f} |ddtg|={ddtg:.6f} |dmetric|max={dmet:.6f} "
              f"success ours/truth={m['success']:.0f}/{t_m.get('success', float('nan')):.0f} {'OK' if ok else 'FAIL'}")
    env.close()
    print(f"\n{len(indices) - len(failed) - len(holes)} ok, {len(failed)} fail, {len(holes)} truth holes; "
          f"worst |dpos| {worst['pos']:.6f} m, |ddtg| {worst['dtg']:.6f} m, |dmetric| {worst['metric']:.6f}")
    if failed:
        print("failed episodes:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
