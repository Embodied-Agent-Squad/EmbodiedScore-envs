"""Fetch the VLNverse release from Hugging Face into the package's data layout.

    python scripts/download_vlnverse_data.py splits                       # -> $EMBODIEDSCORE_DATA_ROOT/vlnverse/raw_data/final_splits
    python scripts/download_vlnverse_data.py scenes --split fine/val_unseen [--limit 5]   # the scenes that split uses
    python scripts/download_vlnverse_data.py scenes --scan kujiale_0272 kujiale_0234      # named scenes
    python scripts/download_vlnverse_data.py check --split fine/val_unseen                # what is missing locally

Splits come from ``Eyz/VLNVerse_data`` (only ``raw_data/final_splits/`` is
fetched — the rest of that dataset is trajectory data for training and is not
needed here), scenes from ``Eyz/VLNVerse_scene`` into
``$EMBODIEDSCORE_SCENE_ROOT/vlnverse/<scan>/`` (one folder per scan; the whole
set is ~300 GB, a single scene ~1 GB). ``freemap.npy`` — the occupancy grid the
kinematics need — is not part of the Hugging Face scene release at the time of
writing; ``check`` reports which scenes lack it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATA_REPO = "Eyz/VLNVerse_data"
SCENE_REPO = "Eyz/VLNVerse_scene"
SPLITS_PREFIX = "raw_data/final_splits"


def _roots(args) -> tuple[Path, Path]:
    """(``<data_root>/vlnverse``, ``<scene_root>/vlnverse``) — the corpus directories."""
    data_root = args.data_root or os.environ.get("EMBODIEDSCORE_DATA_ROOT")
    scene_root = args.scene_root or os.environ.get("EMBODIEDSCORE_SCENE_ROOT")
    if not data_root or not scene_root:
        sys.exit("set EMBODIEDSCORE_DATA_ROOT and EMBODIEDSCORE_SCENE_ROOT (or pass --data-root / --scene-root)")
    os.environ.setdefault("EMBODIEDSCORE_DATA_ROOT", data_root)
    os.environ.setdefault("EMBODIEDSCORE_SCENE_ROOT", scene_root)
    return Path(data_root).expanduser() / "vlnverse", Path(scene_root).expanduser() / "vlnverse"


def _scans_of_split(split: str) -> list[str]:
    granularity, name = split.split("/", 1)
    from embodiedscore_envs.benchmarks.vlnverse import load_episodes
    seen: dict[str, None] = {}
    for e in load_episodes(granularity, name):
        seen.setdefault(e.scene.scan, None)
    return list(seen)


def cmd_splits(args):
    from huggingface_hub import snapshot_download
    data_dir, _ = _roots(args)
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=str(data_dir),
                      allow_patterns=[f"{SPLITS_PREFIX}/*"])
    print(f"splits under {data_dir / SPLITS_PREFIX}")


def cmd_scenes(args):
    from huggingface_hub import snapshot_download
    _, scene_dir = _roots(args)
    scans = list(args.scan or [])
    if args.split:
        scans += _scans_of_split(args.split)
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        sys.exit("nothing to fetch: pass --scan ... or --split <granularity>/<split>")
    snapshot_download(SCENE_REPO, repo_type="dataset", local_dir=str(scene_dir),
                      allow_patterns=[f"{s}/*" for s in scans])
    print(f"{len(scans)} scene(s) under {scene_dir}")


def cmd_check(args):
    _, scene_dir = _roots(args)
    from embodiedscore_envs.benchmarks.env.sim.isaac.scene import IsaacSceneRef
    scans = _scans_of_split(args.split)
    missing_usd, missing_freemap = [], []
    for s in scans:
        ref = IsaacSceneRef(scan=s, scene_dir=str(scene_dir / s))
        try:
            ref.scene_file
        except FileNotFoundError:
            missing_usd.append(s)
        if not os.path.isfile(ref.freemap_file):
            missing_freemap.append(s)
    print(f"{args.split}: {len(scans)} scenes; missing USD: {len(missing_usd)}; missing freemap.npy: {len(missing_freemap)}")
    for label, lst in (("USD", missing_usd), ("freemap", missing_freemap)):
        if lst:
            print(f"  no {label}: {' '.join(lst)}")
    return 1 if (missing_usd or missing_freemap) else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="overrides EMBODIEDSCORE_DATA_ROOT")
    p.add_argument("--scene-root", default=None, help="overrides EMBODIEDSCORE_SCENE_ROOT")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("splits")
    s = sub.add_parser("scenes")
    s.add_argument("--scan", nargs="*")
    s.add_argument("--split", default=None, help="<granularity>/<split>, e.g. fine/val_unseen")
    s.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--split", required=True)
    args = p.parse_args()
    return {"splits": cmd_splits, "scenes": cmd_scenes, "check": cmd_check}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
