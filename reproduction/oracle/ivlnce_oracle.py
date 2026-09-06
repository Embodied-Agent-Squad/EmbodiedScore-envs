"""IVLN-CE replay truth from the legacy stack (habitat-lab 0.1.7 + VLN-CE + the
workspace's env_ivlnce nodeset). Runs in the legacy conda env:

    PYTHONPATH=agentcanvas/backend ~/miniforge3/envs/ac-vlnce/bin/python <this file> \
        --split val_unseen --tour 0 --n 6 --out ivlnce_tour0.json

Walks the first ``n`` episodes of one tour in order with a deterministic
policy — even episodes: the manager's own shortest-path follower to the goal
then STOP; odd episodes: 10 seeded random moves then STOP — letting the
nodeset run the oracle transit between episodes. Records per episode the
agent actions, the position after every action, the seven metrics, the
transit step count and the tour t-nDTW, plus the full tour path at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO_ROOT = "/data/ws_vln/vlnworkspace"
sys.path.insert(0, os.path.join(REPO_ROOT, "workspace", "nodesets", "env"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val_unseen")
    ap.add_argument("--tour", type=int, default=0)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import env_ivlnce as pkg
    mgr = pkg.IvlnceEnvManager.get()
    mgr.initialize(pkg._R2R_CONFIG, args.split, 0, 500)
    eps = mgr._episodes()
    tour = [i for i, e in enumerate(eps) if str(getattr(e, "tour_id", "")) == str(args.tour)][: args.n]
    rng = random.Random(args.seed)
    sim = mgr._env._env.sim
    cases = []
    for k, idx in enumerate(tour):
        meta = mgr.set_episode_by_index(idx)
        actions: list[int] = []
        track = [list(map(float, sim.get_agent_state().position))]

        def do(a: int):
            r = mgr.step(int(a))
            actions.append(int(a))
            if int(a) != 0:
                track.append(list(map(float, sim.get_agent_state().position)))
            return r

        if k % 2 == 0:
            goal = mgr._env._env.current_episode.goals[0].position
            follower = mgr._ensure_follower()
            n = 0
            while n < 480 and not mgr._episode_done:
                try:
                    a = follower.get_next_action(goal)
                except Exception:
                    break
                if a is None or int(a) == 0:
                    break
                do(int(a)); n += 1
        else:
            for _ in range(10):
                if mgr._episode_done:
                    break
                do(rng.choice([1, 2, 3]))
        if not mgr._episode_done:
            do(0)
        ev = mgr.evaluate()
        tm = mgr.tour_metrics()
        cases.append({
            "index": idx, "episode_id": str(meta.get("episode_id")), "scene_id": str(meta.get("scene_id")),
            "actions": actions, "track": track,
            "metrics": {k_: v for k_, v in ev.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
            "tour": {"transit_steps": tm.get("transit_steps"), "tour_t_ndtw": tm.get("tour_t_ndtw"),
                     "tour_index": tm.get("tour_index"), "teleported": bool(mgr._transit_teleported),
                     "delivered_position": track[0], "path_len": len(mgr.get_tour_path())},
        })
        print(f"ep {idx} (tour pos {k}): {len(actions)} actions, success {ev.get('success')}, spl {ev.get('spl')}, "
              f"transit {tm.get('transit_steps')}, t-nDTW {tm.get('tour_t_ndtw')}", flush=True)
    path = mgr.get_tour_path()
    mgr.shutdown()
    json.dump({"benchmark": "ivlnce", "split": args.split, "tour": args.tour, "episodes": cases, "tour_path": path},
              open(args.out, "w"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
