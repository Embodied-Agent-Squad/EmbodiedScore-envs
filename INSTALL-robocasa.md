# Installing the RoboCasa lines

The `robocasa365-*` and `robocasa-*` lines run RoboCasa in process: robosuite
on MuJoCo, rendered headless through EGL. Nothing of it is a pip dependency of
this package — like habitat-sim and LIBERO, it is installed once into the
interpreter that serves these lines. The package imports it lazily: `import
embodiedscore_envs` needs none of it, and even the episode list of a RoboCasa
line loads without it (a kitchen is *generated*, so an episode is four numbers);
the first `reset()` is where it is required.

## 0. Two releases, two environments

| | RoboCasa (v0.2) | RoboCasa365 (v1.0.1) |
|---|---|---|
| paper | Nasiriany et al., RSS 2024, [arXiv:2406.02523](https://arxiv.org/abs/2406.02523) | ICLR 2026, [arXiv:2603.04356](https://arxiv.org/abs/2603.04356) |
| repo | github.com/robocasa/robocasa, tag `v0.2` (commit `756598a`) | the same repo, `main` (tag `v1.0`, version 1.0.1) |
| tasks | 100 (25 atomic + 75 composite) | 365 (65 atomic + 300 composite) |
| pins | python 3.10, numpy 1.23.3, mujoco 3.2.6 | python 3.11, numpy 2.2.5, mujoco 3.3.1, robosuite ≥ 1.5.2 |
| assets | ~5 GB | ~10 GB (~15 GB unpacked) |
| our lines | `robocasa-pnp` · `-doors` · `-drawers` · `-levers` · `-knobs` · `-insertion` · `-buttons` · `-navigate` | `robocasa365-atomic-seen` · `-composite-seen` · `-composite-unseen` |
| env here | `ac-robocasa` | `ac-robocasa365` |

Both are the **same pip distribution name** (`robocasa`), at incompatible pins,
and they renamed the task classes (`PnPCounterToCab` -> `PickPlaceCounterToCabinet`,
`OpenSingleDoor` -> `OpenCabinet`, …). `robocasa/__init__.py` on `main` asserts
`mujoco.__version__ == "3.3.1"` and `numpy.__version__ in ["2.2.5"]`. **They
cannot share one interpreter**: one clone and one conda env each. Neither is on
PyPI; both install from source.

The engine in this package serves both — only `benchmarks/robocasa.py` knows
which release a line belongs to.

## 1. RoboCasa365 (the `robocasa365-*` lines)

```
mamba create -y -n ac-robocasa365 python=3.11
conda activate ac-robocasa365
pip install --no-cache-dir "numpy==2.2.5" "mujoco==3.3.1" "scipy==1.15.3" "numba==0.61.2" \
    gymnasium pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio imageio-ffmpeg \
    h5py lxml hidapi msgpack pytest
```

robosuite and robocasa go in with `--no-deps`, so their own pins do not pull a
different numpy/mujoco back in (robocasa's `install_requires` also lists
`lerobot==0.3.3` and `tianshou`, neither of which any evaluation code path
imports — `lerobot` only in `robocasa/utils/lerobot_utils.py`, `tianshou` only in
`robocasa/scripts/bench_speed.py`):

```
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite && git checkout 5ce6643f          # v1.5.2 + master fixes, 2026
pip install --no-deps -e .

git clone https://github.com/robocasa/robocasa.git
cd robocasa && git checkout 4f8a298            # main, version 1.0.1
pip install --no-deps -e .
python robocasa/scripts/setup_macros.py < /dev/null
yes y | python -m robocasa.scripts.download_kitchen_assets    # ~10 GB, ~15 min
```

The assets land **inside the checkout**, under
`robocasa/models/assets/{textures,generative_textures,fixtures,objects/…}`; put
the clone somewhere with room (here: `/data/ws_vln/coding-agents/data/robocasa/`).

## 2. RoboCasa v0.2 (the `robocasa-*` lines)

**robosuite v1.5.1, not v1.5.2.** `robocasa/__init__.py` at v0.2 asserts
`mujoco == 3.2.6`, `numpy in 1.23.{2,3,5}` and `robosuite >= 1.5.0`; robosuite
v1.5.2 raised its own floor to `mujoco >= 3.3.0`. v1.5.1 still accepts
`mujoco >= 3.2.3`, so it is the only robosuite that satisfies both.

```
mamba create -y -n ac-robocasa python=3.10
conda activate ac-robocasa
pip install --no-cache-dir "numpy==1.23.3" "mujoco==3.2.6" "numba==0.56.4" "scipy==1.10.1" \
    "gymnasium>=1.3" pygame Pillow opencv-python pyyaml pynput tqdm termcolor imageio h5py lxml hidapi \
    msgpack pytest "qpsolvers[quadprog]"

git clone https://github.com/ARISE-Initiative/robosuite.git robosuite-v02
cd robosuite-v02 && git checkout v1.5.1 && pip install --no-deps -e .

git clone https://github.com/robocasa/robocasa.git robocasa-v02
cd robocasa-v02 && git checkout v0.2 && pip install --no-deps -e .
python robocasa/scripts/setup_macros.py < /dev/null
yes y | python robocasa/scripts/download_kitchen_assets.py    # ~5 GB (it skips aigen_objs)
```

`robocasa` v0.2 predates the `PandaOmron` rename; `kitchen.py` carries the shim
that maps its own `PandaMobile` onto robosuite ≥ 1.5's `PandaOmron`, and the
world here asks for `PandaOmron` either way (`RobocasaBody.robot`). robosuite
v1.5.1 warns that `mink` is missing — that is the GR1 whole-body IK controller,
which these lines never build.

## 3. This package, and a check

The package pins `numpy>=1.24,<2` (the habitat lines reproduce their boards on
numpy 1). RoboCasa365 needs numpy 2.2.5, so install it here **without deps** and
add the three runtime imports by hand:

```
pip install --no-deps -e /path/to/EmbodiedScore-envs
pip install --no-deps numpy-quaternion fastdtw msgpack-numpy    # what the loaders import
MUJOCO_GL=egl python -c "
import embodiedscore_envs as es
env = es.make('robocasa365-atomic-seen', 'mini'); obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, info['language'], info['metrics']); env.close()"
```

`--no-deps` on the three matters in the **v0.2** env: installing them normally
pulls numpy 1.26, which breaks `numba` 0.56.4's binary interface and violates
robocasa v0.2's own numpy assert. There, pin the quaternion build too — the
current wheels need numpy ≥ 1.25:

```
pip install --no-deps --no-build-isolation "numpy-quaternion==2022.4.3"
```

`MUJOCO_GL=egl` is set by `RobocasaWorld` when unset; an NVIDIA driver is needed
for EGL. **Every reset generates a kitchen** — RoboCasa samples the fixtures, the
objects and their placements and rebuilds the MuJoCo model — so a reset costs
tens of seconds, not the ~5 s a LIBERO task costs once.

## 4. Tests

```
MUJOCO_GL=egl EMBODIEDSCORE_RUN_ROBOCASA_TESTS=1 python -m pytest tests/test_robocasa_contracts.py
```

Without the variable the file runs only the declaration contracts (no
simulator). The simulator tests pick the line that matches the release installed
in the interpreter; `EMBODIEDSCORE_ROBOCASA_LINE` overrides it. Two of them
check this package's hard-coded tables against whichever release is present, so
a release bump is caught here rather than in a run:
`test_tables_match_the_installed_release` against the per-task horizons of
`robocasa/utils/dataset_registry.py`, and
`test_excluded_scenes_match_the_installed_release` against the
`EXCLUDE_LAYOUTS` / `EXCLUDE_STYLES` each task class declares (nine of
RoboCasa365's composite tasks rule out target scenes, and a scenario pinned to
one of those would leave the env nothing to sample and raise on reset).

Then the end-to-end proof, which needs no model:

```
MUJOCO_GL=egl python scripts/robocasa_scripted_episode.py \
    --line robocasa365-atomic-seen --task CloseBlenderLid --scenario 0
```

(that script lives in the coding-agents repo, not this package.)
