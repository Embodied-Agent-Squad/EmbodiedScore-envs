"""The VLNverse episode loader against the released split files. Needs
``EMBODIEDSCORE_DATA_ROOT`` (no simulator, no GPU)."""

from __future__ import annotations

import gzip
import json
import os

import pytest

if not (os.environ.get("EMBODIEDSCORE_DATA_ROOT") and os.environ.get("EMBODIEDSCORE_SCENE_ROOT")):
    pytest.skip("EMBODIEDSCORE_DATA_ROOT / EMBODIEDSCORE_SCENE_ROOT not set", allow_module_level=True)

import embodiedscore_envs as vv
from embodiedscore_envs.benchmarks.vlnverse import GRANULARITIES, challenge_ids, load_episodes


def _raw(granularity, split):
    root = os.path.join(os.environ["EMBODIEDSCORE_DATA_ROOT"], "vlnverse", "raw_data", "final_splits")
    for name in (f"{granularity}_{split}.json", f"{granularity}_{split}.json.gz"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as f:
                return json.load(f)["episodes"]
    raise FileNotFoundError((granularity, split))


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_val_unseen_schema(granularity):
    eps = load_episodes(granularity, "val_unseen")
    assert len(eps) == len(_raw(granularity, "val_unseen"))
    for i, e in enumerate(eps[:200]):
        assert e.index == i and e.episode_id
        assert e.goal is not None and e.reference_path is not None
        assert tuple(e.goal.position) == e.reference_path[-1]        # the evaluator's NE target == goals.position
        assert e.goal.radius == 3.0
        assert e.start_position[2] == 0.0 and len(e.start_rotation) == 4
        assert e.instruction.strip()
        assert e.scene.scan == e.info["scan"] and e.scene.scene_dir.endswith(e.scene.scan)
        json.dumps(e.as_dict())
    if granularity == "coarse":
        assert set(eps[0].info["instruction_variants"]) >= {"formal", "natural", "casual"}
        natural = load_episodes(granularity, "val_unseen", instruction_type="natural")
        assert natural[0].instruction == eps[0].info["instruction_variants"]["natural"]
    else:
        assert eps[0].info["instruction_variants"] is None


@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_test_and_challenge_have_no_ground_truth(granularity):
    test = load_episodes(granularity, "test")
    assert len(test) == len(_raw(granularity, "test"))
    assert all(e.goal is None and e.reference_path is None for e in test)
    challenge = load_episodes(granularity, "challenge")
    ids = challenge_ids(granularity)
    assert len(challenge) == len(ids) == 150
    assert {e.episode_id for e in challenge} == ids
    assert [e.index for e in challenge] == list(range(len(challenge)))


def test_filters_only_remove():
    full = load_episodes("fine", "val")
    dedup = load_episodes("fine", "val", filter_same_trajectory=True)
    flat = load_episodes("fine", "val", filter_stairs=True)
    assert len(dedup) <= len(full) and len(flat) <= len(full)
    assert len({e.info["trajectory_id"] for e in dedup}) == len(dedup)


def test_declarations_resolve():
    from embodiedscore_envs.benchmarks.presets import bodies, depth
    for name in ("vlnverse-fine", "vlnverse-coarse"):
        b, up = vv.benchmark(name), vv.benchmark(name, "upstream")
        assert b.variant == "standard" and b.line == name and b.max_episode_steps == 500 and b.engine == "isaac"
        assert b.body is bodies.VLNVERSE_STANDARD and b.depth is depth.STANDARD and not b.polar
        assert up.body is bodies.VLNVERSE and up.depth is None and up.polar
        assert b.episodes("val_unseen")[0].index == 0
    with pytest.raises(ValueError):
        load_episodes("fine", "val_seen")     # the evaluator's name for the "val" file is not a file
