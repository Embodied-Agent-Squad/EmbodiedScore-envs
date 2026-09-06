"""HM-EQA (explore-eqa, Ren et al. 2024) and MT-HM3D (MemoryEQA) — embodied
question answering on HM3D-sem scenes. The environment is navigation with a
question attached: the answer is not an environment concept (compared by
the harness), so the episode ends with STOP or the budget.

Data (under ``EMBODIEDSCORE_DATA_ROOT``):
    hmeqa/questions.csv · questions_mip100.csv · scene_init_poses.csv
    mt_hm3d/MT-HM3D-contextual.csv · questions_mip100.csv · scene_init_poses_all.csv
Rows: scene, floor, question, choices, question_formatted, answer (letter),
label (+ MT-HM3D's source_image, target_objects, contextual_descriptions,
enriched_question). The start pose is ``scene_init_poses[f"{scene}_{floor}"]``
(position in habitat metres, ``init_angle`` = yaw about +y); MT-HM3D forces
``_0`` regardless of the question's floor and parses choices with MemoryEQA's
naive split — both benchmark-native quirks, kept. ``episode_id`` = index in
the split. Scenes: ``EMBODIEDSCORE_SCENE_ROOT/hm3dsem/<scene>/<scene[6:]>.basis.glb``
bare (no scene-dataset config), navmesh file next to it (recomputed for a
1.5 m agent only when missing).

Body = explore-eqa cfg/vlm_exp.yaml: RGB-D 640x480 hfov 120 at 1.5 m, camera
pitched -30° at start — explore-eqa pitches the *agent* (yaw ∘ tilt on the
agent state), so the camera swings about the agent's feet: at -30° it sits
0.75 m ahead of and 0.20 m below the nominal 1.5 m mount, and every tilt
action moves it again (``Body.pitch_moves_camera``); the workspace's discrete surface adds 0.25 m / 30° /
tilt 30° clamped to ±60°. Depth leaves raw (metres). Budget
``num_step = int(sqrt(scene_size) * 3)`` with scene_size from the navmesh
bounds (reported in ``info["step_budget"]``); the discrete stack keeps the
harness's 500-step TimeLimit, the pose stack truncates at num_step.

Two stacks per corpus: ``hmeqa`` / ``mthm3d`` (Discrete, actions 0-5) and
``hmeqa-pose`` / ``mthm3d-pose`` (explore-eqa's native free-pose teleport,
``Box[x, z, yaw]`` in the habitat frame — the TSDF planner's z-up "normal"
frame is the caller's convention: normal (x, y) -> habitat (x, z=-y)).
"""

from __future__ import annotations

import csv
import math
import os

from .env import (Act, Benchmark, Body, CameraSpec, Episode, NavMesh, NavMetrics, Question, SceneRef, data_root,
                  scene_root)

MAX_STEP_ROOM_SIZE_RATIO = 3.0

BODY = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=60.0, camera_pitch_deg=-30.0,
            pitch_moves_camera=True, locomotion="teleport", agent_height_m=1.5, agent_radius_m=0.1, allow_sliding=True,
            rgb=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)), depth=CameraSpec(640, 480, 120.0, (0.0, 1.5, 0.0)),
            navmesh=NavMesh.file(fallback=NavMesh.recompute(agent_radius=0.1, agent_height=1.5)))
ACTIONS = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT, Act.LOOK_UP, Act.LOOK_DOWN)
METRIC_KEYS = ("path_length", "steps_taken")

_CORPORA = {
    "hmeqa": {"dir": "hmeqa", "questions": "questions.csv", "init_poses": "scene_init_poses.csv", "force_floor0": False},
    "mthm3d": {"dir": "mt_hm3d", "questions": "MT-HM3D-contextual.csv", "init_poses": "scene_init_poses_all.csv",
               "force_floor0": True},
}


def parse_choices(raw: str) -> list[str]:
    """explore-eqa run_vlm_exp.py:78."""
    return [c.split("'")[1] for c in raw.split("',")]


def parse_choices_memoryeqa(raw: str) -> list[str]:
    """MemoryEQA memory_eqa.py:167 (differs from the above on 38/1587 rows; benchmark-native)."""
    return [c.strip("'\"") for c in raw.strip("[]").split(", ")]


def format_question(question: str, choices: list[str]) -> str:
    """explore-eqa's LLaMA-style A/B/C/D tail (run_vlm_exp.py:84-88)."""
    out = question
    for letter, choice in zip("ABCD", choices):
        out += "\n" + letter + ". " + choice
    return out


def scene_size(world) -> float:
    """explore-eqa's ``scene_size`` — the navmesh bounds' footprint in the z-up
    frame (habitat (x, y, z) -> normal (x, -z, y)), |Δx · Δ(-z)|."""
    lo, hi = world.bounds()
    dx, dy_normal = hi[0] - lo[0], (-hi[2]) - (-lo[2])
    return float(abs(dx * dy_normal))


def step_budget(world, episode: Episode) -> int:
    return int(math.sqrt(scene_size(world)) * MAX_STEP_ROOM_SIZE_RATIO)


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f, skipinitialspace=True)]


def load_episodes(corpus: str, split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    spec = _CORPORA[corpus]
    base = data_root(data_root_) / spec["dir"]
    qfile = base / (spec["questions"] if split == "val" else f"questions_{split}.csv")
    rows = _read_csv(str(qfile))
    poses = {r["scene_floor"]: r for r in _read_csv(str(base / spec["init_poses"]))}
    sroot = scene_root(scene_root_)
    parse = parse_choices_memoryeqa if corpus == "mthm3d" else parse_choices
    out: list[Episode] = []
    for i, row in enumerate(rows):
        scene, floor = row["scene"], row["floor"]
        key = f"{scene}_0" if spec["force_floor0"] else f"{scene}_{floor}"
        init = poses[key]
        yaw = float(init["init_angle"])
        short = scene[6:] if len(scene) > 6 else scene
        choices = parse(row["choices"])
        info = {k: v for k, v in row.items() if k not in ("question", "choices", "answer")}
        info.update({"scene_floor": key, "init_angle": yaw, "question_formatted": format_question(row["question"], choices)})
        out.append(Episode(
            index=len(out), episode_id=str(i),
            scene=SceneRef(scene_file=str(sroot / "hm3dsem" / scene / f"{short}.basis.glb")),
            start_position=(float(init["init_x"]), float(init["init_y"]), float(init["init_z"])),
            start_rotation=(0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)),
            goal=Question(text=row["question"], answer=row["answer"], choices=tuple(choices)),
            info=info,
        ))
    return out


def _decl(corpus: str, pose: bool) -> Benchmark:
    name = corpus + ("-pose" if pose else "")
    label = {"hmeqa": "HMEQA", "mthm3d": "MTHM3D"}[corpus]
    return Benchmark(
        name=name, gym_id=f"EmbodiedScore/{label}{'-Pose' if pose else ''}-v0", body=BODY, actions=ACTIONS,
        splits=("val", "mip100"),
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(corpus, split, data_root, scene_root),
        metrics=lambda env, **o: NavMetrics(env, keys=METRIC_KEYS, path_length_from="euclid64"),
        depth=None, max_episode_steps=None if pose else 500, budget=step_budget, truncate_at_budget=pose,
        pose=pose, pose_snap=False,
        description=f"{label} on HM3D-sem ({'explore-eqa free-pose protocol' if pose else 'discrete surface, 0.25 m / 30° / tilt ±60°'})",
    )


BENCHMARKS = (_decl("hmeqa", False), _decl("hmeqa", True), _decl("mthm3d", False), _decl("mthm3d", True))
