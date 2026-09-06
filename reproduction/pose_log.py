"""HM-EQA free-pose protocol reproduction: replay the explore-eqa canvas graph's
node log (``outputs/eval_runs/<ts>/episodes/ep*/log.jsonl``) through
``hmeqa-pose`` and compare the per-step agent pose, the per-episode budget
(num_step) and floor height, and the truncation step.

    EMBODIEDSCORE_DATA_ROOT=... EMBODIEDSCORE_SCENE_ROOT=... python -u reproduction/pose_log.py \
        --run outputs/eval_runs/20260615_183412 [--benchmark hmeqa-pose --split val]

The legacy action is explore-eqa's z-up "normal" frame (x, y) + yaw; the
gym action is habitat-frame [x, z, yaw] with z = -y.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import embodiedscore_envs as es


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--benchmark", default="hmeqa-pose")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()
    env = es.make(args.benchmark, args.split)
    n_ok = n_fail = 0
    for ep_dir in sorted(Path(args.run).glob("episodes/ep*")):
        reset_out = None
        steps = []
        for line in (ep_dir / "log.jsonl").read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            nt = d.get("node_type", "")
            if nt.endswith("__reset") and reset_out is None:
                reset_out = d.get("outputs") or {}
            elif nt.endswith("__step_pose"):
                a = json.loads(d["inputs"]["action"]) if isinstance(d["inputs"].get("action"), str) else d["inputs"]["action"]
                steps.append((a, (d.get("outputs") or {})))
        if reset_out is None:
            continue
        problems = []
        _, info = env.reset(options={"episode_id": str(reset_out["episode_id"])})
        if int(info["step_budget"]) != int(reset_out["num_step"]):
            problems.append(f"num_step truth {reset_out['num_step']} ours {info['step_budget']}")
        if abs(float(info["floor_y"]) - float(reset_out["floor_height"])) > args.tol:
            problems.append(f"floor_height truth {reset_out['floor_height']} ours {info['floor_y']}")
        max_dpos = 0.0
        truncated_at = None
        for k, (a, out) in enumerate(steps):
            x, y = a["position_normal"]
            _, _, term, trunc, info = env.step([x, -y, a["angle"]])
            truth_pos = (out.get("info") or {}).get("pose", {}).get("position")
            if truth_pos is not None:
                max_dpos = max(max_dpos, float(np.linalg.norm(np.asarray(truth_pos) - np.asarray(info["position"]))))
            if abs(float(info["heading"]) - float(a["angle"])) > 1e-9:
                problems.append(f"heading truth {a['angle']} ours {info['heading']}")
            if trunc and truncated_at is None:
                truncated_at = k + 1
            if bool(out.get("truncated")) != bool(trunc):
                problems.append(f"truncated at step {k + 1}: truth {out.get('truncated')} ours {trunc}")
                break
        if max_dpos > args.tol:
            problems.append(f"|dpos|max {max_dpos:.2e}")
        ok = not problems
        n_ok += ok
        n_fail += (not ok)
        print(f"{ep_dir.name} id={reset_out['episode_id']}: steps={len(steps)} num_step={reset_out['num_step']} "
              f"|dpos|max={max_dpos:.2e} truncated_at={truncated_at} {'OK' if ok else 'FAIL ' + '; '.join(problems)}")
    env.close()
    print(f"\n{n_ok} ok, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
