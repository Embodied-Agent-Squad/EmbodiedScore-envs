"""ObjectNav (habitat-lab 0.2.4 ObjectNav-v1 task) on HM3D v1 / v2 and MP3D v1,
and HM3D-OVON — the same body, actions and measures on the open-vocabulary
episode files (declared here because everything but the files is shared).

Data (under ``EMBODIEDSCORE_DATA_ROOT/objectnav``):
    {hm3d/v1 | hm3d/v2 | mp3d/v1}/{split}/{split}.json.gz              shell (category tables; episodes
                                                                        only when there is no content/)
    {...}/{split}/content/<scene>.json.gz                                one shard per scene, name-sorted
Each shard is ``{episodes, goals_by_category, category_to_*}``; an episode's
goals are ``goals_by_category[f"{basename(scene_id)}_{object_category}"]`` —
every instance of that category in the scene, each with its view points.
``episode_id`` is re-numbered ``str(i)`` within a shard, as habitat-lab does
(unique per scene, not per split); the flat-file case numbers over the file.

Scenes: HM3D ``hm3d/val/<id>/<short>.basis.glb`` with the annotated
scene-dataset config (v2 episodes say ``hm3d_v0.2/{val,minival}`` — provide
those as links to ``hm3d/val``); MP3D ``mp3d/<scan>/<scan>.glb`` bare.
The navmesh is recomputed for the 0.18 m / 0.88 m agent on every scene load
(habitat-lab does so because the shipped files are baked for 0.1 / 1.5).

Body = benchmark/nav/objectnav/objectnav_hm3d.yaml (== objectnav_mp3d.yaml):
0.25 m, turn 30°, tilt 30° unlimited, RGB-D 640x480 hfov 79 at 0.88 m,
depth 0.5-5 m normalised, sliding off, 500 steps. Actions 0-5.
Metrics = distance_to_goal (to the nearest view point) / success (0.1 m) / spl / soft_spl.

OVON (``EMBODIEDSCORE_DATA_ROOT/ovon/hm3d``): three val splits whose directory
and file names differ (upstream's tarball keeps generation-time names) plus
derived ``mip{100,60}_*`` files; ``episode_id`` is kept as authored (the OVON
loader does not renumber); ``children_object_categories`` rides in info and
does not widen the goal set (habitat's stock DistanceToGoal never reads it).
success_distance stays at habitat's 0.1, not upstream OVON's 0.25 — the
workspace nodeset's docstring records why.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from .env import (OBJECTNAV_KEYS, Act, Benchmark, Body, CameraSpec, DepthSpec, Episode, NavMesh, NavMetrics,
                  ObjectGoal, ObjectInstance, SceneRef, data_root, scene_root)

SUCCESS_DISTANCE = 0.1
_SCENE_PREFIX = "data/scene_datasets/"

BODY = Body(forward_step_m=0.25, turn_deg=30.0, tilt_deg=30.0, tilt_limit_deg=None,
            agent_height_m=0.88, agent_radius_m=0.18, allow_sliding=False,
            rgb=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)), depth=CameraSpec(640, 480, 79.0, (0.0, 0.88, 0.0)),
            navmesh=NavMesh.recompute(agent_radius=0.18, agent_height=0.88))
ACTIONS = (Act.STOP, Act.FORWARD, Act.LEFT, Act.RIGHT, Act.LOOK_UP, Act.LOOK_DOWN)
DEPTH = DepthSpec(0.5, 5.0, normalize=True)

_VARIANTS = {"hm3d-v1": "hm3d/v1", "hm3d-v2": "hm3d/v2", "mp3d-v1": "mp3d/v1"}
_OVON_MIP_N = (100, 60)
_OVON_FILES = {"val_seen": "val_seen", "val_seen_synonyms": "val_unseen_easy", "val_unseen": "val_unseen_hard",
               **{f"mip{n}_{suf}": f"mip{n}_{suf}" for n in _OVON_MIP_N for suf in ("seen", "seen_synonyms", "unseen")}}


def _load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _scene_ref(scene_id: str, dataset_config: str | None, sroot: Path) -> SceneRef:
    rel = scene_id[len(_SCENE_PREFIX):] if scene_id.startswith(_SCENE_PREFIX) else scene_id
    rel = rel.replace("//", "/")
    cfg = None
    if dataset_config and dataset_config != "default":
        c = dataset_config[len("./"):] if dataset_config.startswith("./") else dataset_config
        c = c[len(_SCENE_PREFIX):] if c.startswith(_SCENE_PREFIX) else c
        cfg = str(sroot / c)
    return SceneRef(scene_file=str(sroot / rel), dataset_config=cfg)


def parse_instances(goals: list[dict]) -> tuple[ObjectInstance, ...]:
    """habitat-lab's ObjectGoal list -> ObjectInstance tuple (view points = agent positions)."""
    out = []
    for g in goals:
        vps = tuple(tuple(float(x) for x in vp["agent_state"]["position"]) for vp in g.get("view_points", []))
        out.append(ObjectInstance(object_id=str(g.get("object_id")), category=str(g.get("object_category")),
                                  position=tuple(float(x) for x in g["position"]), view_points=vps))
    return tuple(out)


def _episodes_from(raw: dict, sroot: Path, start_index: int, renumber: bool = True) -> list[Episode]:
    gbc = {k: parse_instances(v) for k, v in raw.get("goals_by_category", {}).items()}
    out: list[Episode] = []
    for i, e in enumerate(raw["episodes"]):
        scene_id = e["scene_id"]
        key = f"{os.path.basename(scene_id)}_{e['object_category']}"
        info = dict(e.get("info") or {})
        info.update({"scene_id": scene_id, "object_category": e["object_category"], "goals_key": key,
                     "shortest_paths": e.get("shortest_paths"),
                     "children_object_categories": e.get("children_object_categories")})
        out.append(Episode(
            index=start_index + len(out),
            episode_id=str(i) if renumber else str(e["episode_id"]),
            scene=_scene_ref(scene_id, e.get("scene_dataset_config"), sroot),
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal=ObjectGoal(category=str(e["object_category"]), instances=gbc[key]),
            info=info,
        ))
    return out


def load_split(base: Path, split: str, sroot: Path, renumber: bool = True, shell_name: str | None = None) -> list[Episode]:
    """habitat-lab PointNavDatasetV1 loading order: the shell, then every
    ``content/*.json.gz`` shard sorted by name; without content/, the shell's episodes."""
    shell = base / split / f"{shell_name or split}.json.gz"
    if not shell.is_file():
        shell = base / f"{shell_name or split}.json.gz"
    raw = _load_json_gz(shell)
    content = shell.parent / "content"
    out: list[Episode] = []
    if content.is_dir():
        shards = sorted(p for p in os.listdir(content) if p.endswith(".json.gz") and not p.startswith("._"))
        for name in shards:
            out.extend(_episodes_from(_load_json_gz(content / name), sroot, len(out), renumber))
    else:
        out.extend(_episodes_from(raw, sroot, 0, renumber))
    return out


def load_episodes(variant: str, split: str, data_root_=None, scene_root_=None) -> list[Episode]:
    if variant == "ovon":
        return load_split(data_root(data_root_) / "ovon" / "hm3d", split, scene_root(scene_root_),
                          renumber=False, shell_name=_OVON_FILES.get(split, split))
    return load_split(data_root(data_root_) / "objectnav" / _VARIANTS[variant], split, scene_root(scene_root_))


def _decl(variant: str, gym_id: str, splits: tuple[str, ...], name: str | None = None) -> Benchmark:
    return Benchmark(
        name=name or f"objectnav-{variant}", gym_id=gym_id, body=BODY, actions=ACTIONS, splits=splits,
        episodes=lambda split, data_root=None, scene_root=None, **kw: load_episodes(variant, split, data_root, scene_root),
        metrics=lambda env, **o: NavMetrics(env, success_distance=o.get("success_distance", SUCCESS_DISTANCE), keys=OBJECTNAV_KEYS),
        depth=DEPTH, max_episode_steps=500,
        description=f"ObjectNav {variant} (habitat-lab 0.2.4 ObjectNav-v1 numerics)",
    )


BENCHMARKS = (
    _decl("hm3d-v1", "EmbodiedScore/ObjectNav-HM3Dv1-v0", ("train", "val", "val_mini")),
    _decl("hm3d-v2", "EmbodiedScore/ObjectNav-HM3Dv2-v0", ("train", "val", "val_mini")),
    _decl("mp3d-v1", "EmbodiedScore/ObjectNav-MP3D-v0", ("train", "val", "val_mini")),
    _decl("ovon", "EmbodiedScore/OVON-v0", tuple(_OVON_FILES), name="ovon"),
)
