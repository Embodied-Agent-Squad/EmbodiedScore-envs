"""Replay recorded action sequences through the Gymnasium stack and compare
with what the legacy stack (habitat-lab 0.1.7 / 0.2.4, raw habitat-sim 0.3.3)
reported — the acceptance test of every EmbodiedScore-envs benchmark.

Truth sources (``--mode``):
  vlnce       coding-agent std board run: episode_<i>.jsonl transcripts + topdown_truth/ep<i>.json
              (per-step track, final dtg, seven metrics from the 0.1.7 evaluator)
  transcript  coding-agent run: episode_<i>.jsonl transcripts + their terminal episode_metrics line
              (ObjectNav / OVON / EXPRESS: terminal metrics only)
  actions-log coding-agent run with live_<i>/actions.log batches (HM-EQA / MT-HM3D: per-batch
              camera_tilt_deg + terminal metrics)
  oracle      a JSON written by scripts/oracle/*.py in the legacy env: per episode the action
              list, the per-step track and the terminal metrics (GOAT, IVLN-CE)

For every episode: reset(options={"episode": i}) (checked against the recorded
episode_id), replay the actions, then compare per-step positions (when the
truth has them), final distance_to_goal and every metric key both sides have.
Exit 1 if any episode exceeds the tolerances.

Usage:
    EMBODIEDSCORE_DATA_ROOT=... EMBODIEDSCORE_SCENE_ROOT=... python -u scripts/parity_replay.py \
        --benchmark objectnav-hm3d-v1 --split val --mode transcript --run <run_dir> [--eps 0-99]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import embodiedscore_envs as es
from embodiedscore_envs.benchmarks.env import Act

SPLIT_SUFFIXES = ("_en", "_hi", "_te")   # board display split -> loader split
# legacy metric name -> ours ("@x" = a reset-info fact, not a metric)
ALIASES = {"path_len": "path_length", "d_t": "distance_to_goal", "num_steps": "@step_budget",
           "gt_geodesic": "@episode.info.geodesic_distance"}


# ---- truth readers -----------------------------------------------------------

def transcript_actions(run_dir: Path, index: int) -> list[int]:
    """Every action the episode actually executed, in order, STOP included
    (the transcript's step tool carries a list; its result says how many ran)."""
    out: list[int] = []
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
                out.append(a)
                if a == 0:
                    return out
    return out


def transcript_terminal_metrics(run_dir: Path, index: int) -> dict[str, Any] | None:
    for line in (run_dir / f"episode_{index}.jsonl").read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("kind") == "episode_metrics":
            return d.get("metrics") or {}
    return None


def actions_log(run_dir: Path, index: int) -> list[dict[str, Any]]:
    p = run_dir / f"live_{index}" / "actions.log"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def parse_eps(spec: str) -> list[int]:
    idx: list[int] = []
    for tok in spec.split(","):
        a, _, b = tok.partition("-")
        idx.extend(range(int(a), int(b or a) + 1))
    return idx


def cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    """[{index, episode_id, actions, track?, dtg?, metrics?, extra?}, ...]"""
    out: list[dict[str, Any]] = []
    if args.mode == "oracle":
        truth = json.loads(Path(args.truth).read_text())
        for c in truth["episodes"]:
            out.append(c)
        return out
    run = Path(args.run).resolve()
    summary = json.loads((run / "summary.json").read_text())
    by_index = {e["index"]: e for e in summary["episodes"]}
    for i in parse_eps(args.eps):
        if i not in by_index or not (run / f"episode_{i}.jsonl").is_file():
            continue
        rec = by_index[i]
        case: dict[str, Any] = {"index": i, "episode_id": str(rec["episode_id"]), "scene_id": rec.get("scene_id")}
        if args.mode == "vlnce":
            tpath = run / "topdown_truth" / f"ep{i}.json"
            if not tpath.is_file():
                continue
            t = json.loads(tpath.read_text())
            case.update(actions=transcript_actions(run, i), track=t["track"], dtg=t.get("dtg"), metrics=t.get("metrics"))
        elif args.mode == "transcript":
            case.update(actions=transcript_actions(run, i), metrics=transcript_terminal_metrics(run, i) or rec.get("metrics"))
        elif args.mode == "actions-log":
            batches = actions_log(run, i)
            # a live dir accumulates every attempt at the episode (the driver retries);
            # keep the last attempt = the tail after the last steps_taken_total reset
            attempts = [[]]
            prev = -1
            for b_ in batches:
                s_ = b_.get("steps_taken_total")
                ex_ = int(b_.get("executed", 0) or 0)
                # a new attempt starts where the running total drops, or where a batch's
                # total equals its own executed count although steps were already taken
                if s_ is not None and (s_ < prev or (ex_ > 0 and s_ == ex_ and prev > 0)):
                    attempts.append([])
                if s_ is not None:
                    prev = s_
                attempts[-1].append(b_)
            restarted = len(attempts) > 1
            batches = attempts[-1]
            acts: list[int] = []
            tilts: list[float | None] = []
            for b in batches:
                n = int(b.get("executed", len(b.get("actions", []))))
                acts.extend(int(a) for a in b.get("actions", [])[:n])
                tilts.append(b.get("camera_tilt_deg"))
            frames = {}
            live = run / f"live_{i}"
            if live.is_dir() and not restarted:     # frames of retried episodes are ambiguous
                for f in live.iterdir():
                    m_ = re.match(r"obs_\d+_step(\d+)\.png$", f.name)
                    if m_:
                        frames[int(m_.group(1))] = str(f)
            case.update(actions=acts, tilt_after_batch=tilts, batch_sizes=[int(b.get("executed", 0)) for b in batches],
                        frames=frames, metrics=transcript_terminal_metrics(run, i) or rec.get("metrics"))
        out.append(case)
    return out


# ---- replay -------------------------------------------------------------------

def _num(x: Any) -> float:
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _close(a: float, b: float, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--mode", choices=("vlnce", "transcript", "actions-log", "oracle"), required=True)
    ap.add_argument("--run", help="coding-agent run directory (vlnce / transcript / actions-log)")
    ap.add_argument("--truth", help="oracle JSON (oracle mode)")
    ap.add_argument("--eps", default="0-99")
    ap.add_argument("--tol-pos", type=float, default=0.0)
    ap.add_argument("--tol-metric", type=float, default=1e-6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--loader", default="{}", help="JSON kwargs for the episode loader")
    args = ap.parse_args()

    env = es.make(args.benchmark, args.split, gpu_id=args.gpu, **json.loads(args.loader))
    table = env.unwrapped.actions
    todo = cases(args)
    worst = {"pos": 0.0, "metric": 0.0}
    failed: list[int] = []
    holes: list[int] = []
    n_ok = 0
    # Episode identity: legacy habitat-lab nodesets address a seed-100-shuffled
    # list, so the recorded index is not ours — select by episode_id (unique in
    # flat splits) or by (scene, episode_id) when ids repeat across scenes.
    import os
    eps = env.unwrapped.episodes
    by_id = {}
    for e in eps:
        by_id.setdefault(e.episode_id, []).append(e.index)
    by_scene_id = {(os.path.basename(e.scene.scene_file), e.episode_id): e.index for e in eps}

    def locate(c):
        if args.mode == "oracle" and "index" in c and (c.get("by_index") or args.benchmark == "ivlnce"):
            return int(c["index"])
        eid = str(c.get("episode_id"))
        cand = by_id.get(eid, [])
        if len(cand) == 1:
            return cand[0]
        scene = os.path.basename(str(c.get("scene_id") or "")).replace("//", "/")
        return by_scene_id.get((scene, eid))

    for c in todo:
        i = int(c["index"])
        target = locate(c)
        if target is None:
            print(f"ep{i:3d}: cannot locate episode {c.get('episode_id')} ({c.get('scene_id')}) — FAIL")
            failed.append(i)
            continue
        obs, info = env.reset(options={"episode": target})
        reset_info = info
        got_id = info["episode"]["episode_id"]
        frames = c.get("frames") or {}
        frame_diffs: list[float] = []

        def frame_check(n_steps: int, rgb) -> None:
            if n_steps in frames:
                from PIL import Image
                ref = np.asarray(Image.open(frames[n_steps]).convert("RGB"), dtype=np.float32)
                if ref.shape == rgb.shape:
                    frame_diffs.append(float(np.abs(ref - rgb.astype(np.float32)).mean()))
        frame_check(0, obs["rgb"])
        track = [info["position"]]
        tilts: list[float] = []
        batch_sizes = c.get("batch_sizes")
        n_done = 0
        term = trunc = False
        actions = list(c.get("actions", []))
        batches = batch_sizes or [len(actions)]
        pos_in = 0
        for size in batches:
            for a in actions[pos_in: pos_in + size]:
                if term or trunc:
                    break
                if Act(a) not in table:
                    print(f"ep{i:3d}: action {a} not in table {table} — skipped action")
                    continue
                obs, _, term, trunc, info = env.step(a)
                n_done += 1
                frame_check(n_done, obs["rgb"])
                if a != 0:                      # truth tracks record the pose after every *movement*
                    track.append(info["position"])
            pos_in += size
            if batch_sizes:
                tilts.append(info["pitch"])
        m = info.get("metrics", {})

        problems: list[str] = []
        dpos = 0.0
        if c.get("track"):
            tt = np.asarray(c["track"], dtype=np.float64)
            tr = np.asarray(track, dtype=np.float64)
            if len(tt) != len(tr):
                problems.append(f"track length truth {len(tt)} ours {len(tr)}")
                dpos = math.inf
            else:
                dpos = float(np.linalg.norm(tt - tr, axis=1).max()) if len(tt) else 0.0
            worst["pos"] = max(worst["pos"], dpos if math.isfinite(dpos) else worst["pos"])
            if dpos > args.tol_pos:
                problems.append(f"|dpos|max={dpos:.6f}")
        if c.get("tilt_after_batch"):
            want = [t for t in c["tilt_after_batch"] if t is not None]
            got = tilts[: len(want)]
            if len(got) != len(want) or any(abs(float(w) - g) > 1e-3 for w, g in zip(want, got)):
                problems.append(f"tilt sequence differs (truth {want[:6]}… ours {[round(g, 3) for g in got[:6]]}…)")
        if c.get("tour"):
            tt_ = c["tour"]; ot_ = info.get("tour", {})
            if int(tt_.get("transit_steps", -1)) != int(ot_.get("transit_steps", -2)):
                problems.append(f"transit_steps: truth {tt_.get('transit_steps')} ours {ot_.get('transit_steps')}")
            a_, b_ = _num(tt_.get("tour_t_ndtw")), _num(ot_.get("tour_t_ndtw"))
            if not _close(a_, b_, 1e-4):
                problems.append(f"tour_t_ndtw: truth {tt_.get('tour_t_ndtw')} ours {ot_.get('tour_t_ndtw')}")
            d0_ = np.asarray(tt_.get("delivered_position", reset_info["position"]), dtype=np.float64)
            dd_ = float(np.linalg.norm(d0_ - np.asarray(reset_info["position"], dtype=np.float64)))
            if dd_ > args.tol_pos:
                problems.append(f"delivered pose off by {dd_:.6f} m")
        t_m = dict(c.get("metrics") or {})
        if "dtg" in c and c["dtg"] is not None:
            t_m.setdefault("distance_to_goal", c["dtg"])
        if not t_m:
            holes.append(i)
            print(f"ep{i:3d} id={got_id:>6}: no truth metrics — hole")
            continue
        dmet = 0.0
        for k, tv in t_m.items():
            key = ALIASES.get(k, k)
            if key.startswith("@"):
                node = reset_info
                for part in key[1:].split("."):
                    node = node.get(part) if isinstance(node, dict) else None
                if node is None:
                    continue
                ov, tvv = node, tv
            elif key in m:
                ov, tvv = m[key], tv
            else:
                continue
            if k == "d_t" and float(t_m.get("d_t_valid", 1.0)) == 0.0:
                tvv = math.inf          # legacy encodes an unreachable goal as d_t = -1, d_t_valid = 0
            if isinstance(ov, list) or isinstance(tvv, list):
                ov_l, tv_l = [_num(x) for x in (ov or [])], [_num(x) for x in (tvv or [])]
                if len(ov_l) != len(tv_l) or any(not _close(x, y, args.tol_metric) for x, y in zip(ov_l, tv_l)):
                    problems.append(f"{k}: truth {tvv} ours {ov}")
                continue
            a_, b_ = _num(ov), _num(tvv)
            if not _close(a_, b_, args.tol_metric):
                problems.append(f"{k}: truth {tvv} ours {ov}")
            if math.isfinite(a_) and math.isfinite(b_):
                dmet = max(dmet, abs(a_ - b_))
        worst["metric"] = max(worst["metric"], dmet)
        if not math.isfinite(_num(t_m.get("distance_to_goal", 0.0))):
            holes.append(i)
        ok = not problems
        n_ok += int(ok)
        if not ok:
            failed.append(i)
        succ = m.get("success", float("nan"))
        fr = f" |dframe|mean_max={max(frame_diffs):.2f}/255 ({len(frame_diffs)} frames)" if frame_diffs else ""
        print(f"ep{i:3d} id={got_id:>6}: steps={len(c.get('actions', [])):3d} |dpos|max={dpos:.6f} "
              f"|dmetric|max={dmet:.2e} success={succ if isinstance(succ, float) else succ}{fr} "
              f"{'OK' if ok else 'FAIL ' + '; '.join(problems)}")
    if args.mode == "oracle":
        truth_path = json.loads(Path(args.truth).read_text()).get("tour_path")
        if truth_path is not None:
            tw = env
            while not hasattr(tw, "tour_path") and hasattr(tw, "env"):
                tw = tw.env
            ours_path = tw.tour_path if hasattr(tw, "tour_path") else []
            same = len(ours_path) == len(truth_path) and all(
                a["phase"] == b["phase"] and str(a["episode_id"]) == str(b["episode_id"])
                and np.allclose(a["position"], b["position"], atol=args.tol_pos, rtol=0)
                for a, b in zip(ours_path, truth_path))
            print(f"tour path: truth {len(truth_path)} entries, ours {len(ours_path)} — {'IDENTICAL' if same else 'DIFFERENT'}")
            if not same:
                failed.append(-1)
    env.close()
    print(f"\n{n_ok} ok, {len(failed)} fail, {len(holes)} holes (truth dtg inf / no metrics) of {len(todo)}; "
          f"worst |dpos| {worst['pos']:.6f} m, |dmetric| {worst['metric']:.2e}")
    if failed:
        print("failed episodes:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
