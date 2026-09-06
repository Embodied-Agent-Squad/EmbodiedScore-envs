"""VLN-CE episode data: R2R-CE (R2R_VLNCE_v1-3_preprocessed) and RxR-CE
(RxR_VLNCE_v0, guide role), read straight from the released json.gz files.

Layout under ``data_root`` (the directory VLN-CE calls data/datasets):
    R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz        episodes
    R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz     reference locations (nDTW)
    RxR_VLNCE_v0/{split}/{split}_guide.json.gz
    RxR_VLNCE_v0/{split}/{split}_guide_gt.json.gz
Scenes under ``scene_root``: ``{scene_root}/{scene_id}`` where scene_id is the
dataset's ``mp3d/<scan>/<scan>.glb``; the navmesh sits next to it as ``.navmesh``.

Defaults come from EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DATASETS = {"r2r", "rxr"}
_R2R_DIR = "R2R_VLNCE_v1-3_preprocessed"
_RXR_DIR = "RxR_VLNCE_v0"


def data_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_DATA_ROOT")
    if not p:
        raise ValueError("data_root not given and EMBODIEDSCORE_DATA_ROOT unset")
    return Path(p)


def scene_root(explicit: str | os.PathLike | None = None) -> Path:
    p = explicit or os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
    if not p:
        raise ValueError("scene_root not given and EMBODIEDSCORE_SCENE_ROOT unset")
    return Path(p)


@dataclass(frozen=True)
class Episode:
    index: int                      # position in the loaded (filtered) list
    episode_id: str
    scene_id: str                   # e.g. "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
    start_position: tuple[float, float, float]
    start_rotation: tuple[float, float, float, float]   # x, y, z, w
    goal_position: tuple[float, float, float]
    goal_radius: float
    instruction: str
    reference_path: tuple[tuple[float, float, float], ...]
    trajectory_id: str
    language: str | None = None     # RxR only
    instruction_id: str | None = None   # RxR only
    info: dict[str, Any] = field(default_factory=dict)   # dataset's own extras (e.g. geodesic_distance)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _episode_file(dataset: str, split: str, root: Path) -> Path:
    if dataset == "r2r":
        return root / _R2R_DIR / split / f"{split}.json.gz"
    return root / _RXR_DIR / split / f"{split}_guide.json.gz"


def gt_file(dataset: str, split: str, root: Path) -> Path:
    if dataset == "r2r":
        return root / _R2R_DIR / split / f"{split}_gt.json.gz"
    return root / _RXR_DIR / split / f"{split}_guide_gt.json.gz"


def load_episodes(dataset: str, split: str, root: str | os.PathLike | None = None,
                  languages: Iterable[str] | None = ("en-US", "en-IN")) -> list[Episode]:
    """Episodes in file order (this is the order the boards index by).
    ``languages`` filters RxR by ``instruction.language``; ignored for R2R."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}, got {dataset!r}")
    path = _episode_file(dataset, split, data_root(root))
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)["episodes"]
    keep = set(languages) if (dataset == "rxr" and languages) else None
    out: list[Episode] = []
    for e in raw:
        ins = e["instruction"]
        lang = ins.get("language")
        if keep is not None and lang not in keep:
            continue
        goal = e["goals"][0]
        out.append(Episode(
            index=len(out),
            episode_id=str(e["episode_id"]),
            scene_id=e["scene_id"],
            start_position=tuple(float(x) for x in e["start_position"]),
            start_rotation=tuple(float(x) for x in e["start_rotation"]),
            goal_position=tuple(float(x) for x in goal["position"]),
            goal_radius=float(goal.get("radius", 3.0)),
            instruction=ins["instruction_text"],
            reference_path=tuple(tuple(float(x) for x in p) for p in e.get("reference_path", [])),
            trajectory_id=str(e.get("trajectory_id", "")),
            language=lang,
            instruction_id=str(ins["instruction_id"]) if "instruction_id" in ins else None,
            info=dict(e.get("info") or {}),
        ))
    return out


def load_gt_locations(dataset: str, split: str, root: str | os.PathLike | None = None) -> dict[str, list[list[float]]]:
    """episode_id -> reference locations (the nDTW ground truth)."""
    with gzip.open(gt_file(dataset, split, data_root(root)), "rt", encoding="utf-8") as f:
        g = json.load(f)
    return {str(k): v["locations"] for k, v in g.items()}


def scene_paths(ep: Episode, root: str | os.PathLike | None = None) -> tuple[str, str]:
    """(scene .glb, navmesh) absolute paths for an episode."""
    glb = scene_root(root) / ep.scene_id
    return str(glb), str(glb.with_suffix(".navmesh"))
