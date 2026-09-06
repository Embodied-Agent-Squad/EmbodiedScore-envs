"""GOAT-Bench replay truth from the legacy stack (habitat-lab 0.2.4 + the
workspace's env_goat nodeset). Runs in the legacy conda env, NOT in ac-es:

    PYTHONPATH=agentcanvas/backend \
      ~/miniforge3/envs/ac-objnav/bin/python <this file> --split val_unseen --eps 0-11 --out goat_val_unseen.json

For every episode it drives the old manager with a deterministic policy —
sub-goals alternate between the habitat ShortestPathFollower (radius 0.25 m,
toward the geodesically nearest view point, at most 400 steps) and 20 seeded
random actions — closing each with SUBTASK_STOP, and records the action list,
the agent position after every action, and the flattened terminal metrics.
The new stack replays the recorded actions; the policy itself is not replayed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

REPO_ROOT = "/data/ws_vln/vlnworkspace"
sys.path.insert(0, os.path.join(REPO_ROOT, "workspace", "nodesets", "env"))
os.chdir(REPO_ROOT)

FOLLOWER_RADIUS = 0.25
MAX_STEPS_PER_SUBTASK = 400
RANDOM_STEPS = 20


def parse_eps(spec: str) -> list[int]:
    idx: list[int] = []
    for tok in spec.split(","):
        a, _, b = tok.partition("-")
        idx.extend(range(int(a), int(b or a) + 1))
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--eps", default="0-11")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from env_goat import GoatEnvManager
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    mgr = GoatEnvManager.get()
    mgr.initialize(split=args.split, gpu_id=0, max_steps=5000)
    rng = random.Random(args.seed)
    cases = []
    for index in parse_eps(args.eps):
        info = mgr.set_episode_by_index(index)
        ep = mgr._env.current_episode
        sim = mgr._env.sim
        follower = ShortestPathFollower(sim, goal_radius=FOLLOWER_RADIUS, return_one_hot=False)
        actions: list[int] = []
        track = [list(map(float, sim.get_agent_state().position))]
        events = []

        def do(a: int):
            out = mgr.step(int(a))
            actions.append(int(a))
            track.append(list(map(float, sim.get_agent_state().position)))
            if out["info"].get("subtask_advanced"):
                events.append({"step": len(actions), "subtask_success": out["info"].get("subtask_success")})
            return out

        n = len(ep.tasks)
        for subtask in range(n):
            if mgr._episode_done:
                break
            if subtask % 2 == 0:
                vps = [vp["agent_state"]["position"] for g in ep.goals[subtask] for vp in g["view_points"]]
                here = np.asarray(sim.get_agent_state().position, dtype=np.float32)
                best, best_d = None, float("inf")
                for vp in vps:
                    d = sim.geodesic_distance(here, [vp])
                    if d is not None and np.isfinite(d) and d < best_d:
                        best, best_d = vp, float(d)
                steps = 0
                while best is not None and steps < MAX_STEPS_PER_SUBTASK and not mgr._episode_done:
                    try:
                        a = follower.get_next_action(np.asarray(best, dtype=np.float32))
                    except Exception:
                        a = None
                    if a is None or int(a) == 0:
                        break
                    do(int(a)); steps += 1
            else:
                for _ in range(RANDOM_STEPS):
                    if mgr._episode_done:
                        break
                    do(rng.choice([1, 2, 3, 4, 5]))
            if not mgr._episode_done:
                do(6)
        if not mgr._episode_done and index % 3 == 0:
            do(0)   # some episodes end with an explicit STOP (task_success needs it)
        metrics = mgr.evaluate()
        cases.append({"index": index, "episode_id": str(info["episode_id"]), "scene_id": str(info.get("scene_id")),
                      "actions": actions, "track": track, "events": events,
                      "metrics": {k: v for k, v in metrics.items() if not isinstance(v, str)}})
        print(f"ep {index}: {len(actions)} actions, success {metrics.get('success')} spl {metrics.get('spl')}", flush=True)
    mgr.shutdown()
    json.dump({"benchmark": "goat", "split": args.split, "episodes": cases}, open(args.out, "w"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
