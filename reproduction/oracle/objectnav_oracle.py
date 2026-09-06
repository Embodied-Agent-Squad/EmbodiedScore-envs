"""ObjectNav / OVON replay truth from the legacy stack (habitat-lab 0.2.4 +
the workspace's env_objnav / env_ovon nodeset). Runs in the legacy conda env:

    PYTHONPATH=agentcanvas/backend ~/miniforge3/envs/ac-objnav/bin/python <this file> \
        --nodeset objnav --dataset hm3d_v1 --split mip100 --run <coding-agent run dir> --eps 87,96 --out truth.json

Replays each transcript's executed actions through the old manager (the same
index the board used — the manager addresses habitat's seed-100-shuffled
list) and records the action list, the position after every action and the
terminal metrics, in the replay script's oracle format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = "/data/ws_vln/vlnworkspace"
sys.path.insert(0, os.path.join(REPO_ROOT, "workspace", "nodesets", "env"))
sys.path.insert(0, os.path.join(REPO_ROOT, "..", "..", "git_all", "EmbodiedScore-envs", "scripts"))
os.chdir(REPO_ROOT)


def transcript_actions(run_dir: str, index: int) -> list[int]:
    out, pending = [], {}
    for line in open(os.path.join(run_dir, f"episode_{index}.jsonl")):
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


def parse_eps(spec: str) -> list[int]:
    idx = []
    for tok in spec.split(","):
        a, _, b = tok.partition("-")
        idx.extend(range(int(a), int(b or a) + 1))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodeset", choices=("objnav", "ovon"), default="objnav")
    ap.add_argument("--dataset", default="hm3d_v1")
    ap.add_argument("--split", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eps", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.nodeset == "objnav":
        from env_objnav import ObjnavEnvManager as M
        mgr = M.get()
        mgr.initialize(dataset=args.dataset, split=args.split, gpu_id=0, max_steps=500)
    else:
        from env_ovon import OvonEnvManager as M
        mgr = M.get()
        mgr.initialize(split=args.split, gpu_id=0, max_steps=500)
    cases = []
    for i in parse_eps(args.eps):
        info = mgr.set_episode_by_index(i)
        sim = mgr._env.sim
        actions = transcript_actions(args.run, i)
        track = [list(map(float, sim.get_agent_state().position))]
        collided = []
        dtg_steps = [float(mgr._env.get_metrics()["distance_to_goal"])]
        for a in actions:
            out = mgr.step(int(a))
            dtg_steps.append(float(mgr._env.get_metrics()["distance_to_goal"]))
            if int(a) != 0:      # a track point per movement, as the replay script expects
                track.append(list(map(float, sim.get_agent_state().position)))
            collided.append(bool((out.get("info") or {}).get("collided", False)))
            if out.get("terminated") or out.get("truncated"):
                break
        metrics = mgr.evaluate()
        ep = mgr._env.current_episode
        cases.append({"index": i, "episode_id": str(ep.episode_id), "scene_id": str(ep.scene_id), "actions": actions,
                      "track": track, "dtg_steps": dtg_steps,
                      "metrics": {k: v for k, v in metrics.items() if not isinstance(v, str)}})
        print(f"ep {i}: id {ep.episode_id} {len(actions)} actions dtg {metrics.get('distance_to_goal')}", flush=True)
    mgr.shutdown()
    json.dump({"benchmark": args.nodeset, "split": args.split, "episodes": cases}, open(args.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
