"""EXPRESS-Bench (Jiang et al., ICCV 2025) — exploration-aware open-vocabulary
EQA on HM3D-sem. The environment is navigation with a question and a target
place attached; the GPT judge and the c / c* / e_path folding live in the
harness. Environment-side terms: ``path_length`` (pose protocol: geodesic
between consecutive poses; discrete surface: euclidean displacement),
``distance_to_goal`` = d_T (geodesic to ``goal_position``, inf when
unreachable), and the record's ``geodesic_distance`` (l_i) in ``info``.

Data: ``EMBODIEDSCORE_DATA_ROOT/express_bench/express-bench.json`` — 2044
records (episode_id == array index), split by ``scene_id`` prefix into
val 409 / train 1635; ``express-bench_mip100.json`` is a derived file mapped
back through the episode_id identity (refused on any question-text mismatch).
Scenes: ``hm3dsem/<scene>/<scene[6:]>.basis.glb`` bare, navmesh file next to it.

Body = fine_eqa.yaml: RGB-D 512x512 hfov 90 at 1.5 m, camera level; the
workspace's discrete surface adds 0.25 m / 30° (no tilt). Depth raw (metres).
Budget ``num_step = int(sqrt(scene_size) * 3)``; the discrete stack keeps the
harness's 500-step TimeLimit, the pose stack truncates at num_step and snaps
targets to the navmesh (``snap_point``, NaN -> random navigable point within
3 m of the previous pose), as upstream main.py does.
"""

from __future__ import annotations

import json
import math

from .env import (Act, Benchmark, Body, CameraSpec, Episode, NavMesh, NavMetrics, PointGoal, Question, SceneRef,
                  data_root, scene_root)

MAX_STEP_ROOM_SIZE_RATIO = 3.0

BODY = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=None, camera_pitch_deg=0.0,
            locomotion="teleport", agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
            rgb=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)), depth=CameraSpec(512, 512, 90.0, (0.0, 1.5, 0.0)),
            navmesh=NavMesh.file(fallback=NavMesh.recompute(agent_radius=0.1, agent_height=1.5)))
ACTIONS = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT)
METRIC_KEYS = ("distance_to_goal", "path_length", "steps_taken")


def scene_size(world) -> float:
    lo, hi = world.bounds()
    return float(abs((hi[0] - lo[0]) * ((-hi[2]) - (-lo[2]))))


def step_budget(world, episode: Episode) -> int:
    return int(math.sqrt(scene_size(world)) * MAX_STEP_ROOM_SIZE_RATIO)


def _record_split(rec: dict) -> str:
    return rec["scene_id"].split("/")[1]


def load_episodes(split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    base = data_root(data_root_) / "express_bench"
    with open(base / "express-bench.json") as f:
        records = json.load(f)
    if split == "all":
        chosen = list(range(len(records)))
    elif split in ("val", "train"):
        chosen = [i for i, r in enumerate(records) if _record_split(r) == split]
    else:
        with open(base / f"express-bench_{split}.json") as f:
            derived = json.load(f)
        chosen = []
        for d in derived:
            i = int(d["episode_id"])
            if records[i]["question"] != d["question"]:
                raise ValueError(f"express-bench_{split}.json: episode {i} does not match the corpus")
            chosen.append(i)
    sroot = scene_root(scene_root_)
    out: list[Episode] = []
    for i in chosen:
        r = records[i]
        scene = r["scene_id"].split("/")[-1]
        short = scene[6:] if len(scene) > 6 else scene
        w, x, y, z = (float(v) for v in r["start_rotation"])      # stored w-first (np.quaternion order)
        info = {k: v for k, v in r.items() if k not in ("question", "answer", "start_position", "start_rotation", "actions")}
        info.update({"scene": scene, "gt_step_length": r.get("step_length"), "gt_actions": r.get("actions")})
        out.append(Episode(
            index=len(out), episode_id=str(r["episode_id"]),
            scene=SceneRef(scene_file=str(sroot / "hm3dsem" / scene / f"{short}.basis.glb")),
            start_position=tuple(float(v) for v in r["start_position"]),
            start_rotation=(x, y, z, w),
            goal=Question(text=r["question"], answer=r["answer"],
                          target=PointGoal(tuple(float(v) for v in r["goal_position"]), radius=0.0)),
            info=info,
        ))
    return out


def _decl(pose: bool) -> Benchmark:
    return Benchmark(
        name="express-pose" if pose else "express", gym_id=f"EmbodiedScore/EXPRESS{'-Pose' if pose else ''}-v0",
        body=BODY, actions=ACTIONS, splits=("val", "train", "all", "mip100"),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(split, data_root, scene_root),
        metrics=lambda env, **o: NavMetrics(env, keys=METRIC_KEYS, path_length_from="step_geodesic" if pose else "euclid64"),
        depth=None, max_episode_steps=None if pose else 500, budget=step_budget, truncate_at_budget=pose,
        pose=pose, pose_snap=pose,
        description="EXPRESS-Bench on HM3D-sem (" + ("free-pose protocol with navmesh snap" if pose else "discrete surface 0.25 m / 30°") + ")",
    )


BENCHMARKS = (_decl(False), _decl(True))
