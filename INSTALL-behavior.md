# BEHAVIOR-1K for the `behavior-1k` line

Only the `behavior-*` lines need this. End to end: the conda environment,
BEHAVIOR-1K v3.7.2 (OmniGibson 3.7.2 + BDDL + JoyLo) on Isaac Sim 4.5.0, the
~36 GB of assets, and the checks that tell you each layer works. Everything
below was run on Ubuntu 20.04 (glibc 2.31) with an RTX 3090 and driver
570.133.07 — and every deviation from upstream's own `setup.sh` that this box
forced is written down.

**This is not the Isaac Sim the VLNverse lines use.** Those run Isaac Sim 5.1
through a container (INSTALL-isaac.md). BEHAVIOR-1K v3.7.2 pins Isaac Sim
**4.5.0** by a hard assert — `omnigibson/simulator.py` L195,
`assert isaac_version_tuple in m.KIT_FILES` with
`m.KIT_FILES = {(4, 5, 0): "omnigibson_4_5_0.kit"}` (L72-74) — so the two
cannot share an interpreter. Keep them in separate conda environments; they
coexist happily on one box.

## 1. Requirements

| | BEHAVIOR-1K v3.7.2's requirement | what this package was run on |
|---|---|---|
| OS | Ubuntu 22.04+ implied by the manylinux_2_34 wheels | Ubuntu 20.04, glibc 2.31 — `setup.sh`'s `check_glibc_old` rename handles it and Isaac Sim 4.5 **does** boot |
| Python | exactly 3.10 (`setup.sh` L248 aborts otherwise) | 3.10.21 |
| Isaac Sim | 4.5.0, the 26 pip wheels from `pypi.nvidia.com` | same |
| driver / GPU | RTX with RT cores | RTX 3090, driver 570.133.07 (CUDA 12.8) |
| torch | 2.6.0 + cu124 (`setup.sh` L231) | same — see § 3, this is the one pin you must restore by hand |
| disk | ~16 GB for the environment + ~36 GB of assets | 52 GB total |

## 2. The environment and the code

`~` is on the small disk on this box, so the environment lives on `/data` and
is reached through a symlink — `env.python: ac-behavior` in an experiment file
resolves `<conda root>/envs/ac-behavior/bin/python` (`core/envserver.py`
`resolve_python`), and a symlink satisfies it.

```bash
BH=/data/ws_vln/coding-agents/data/behavior
git clone --depth 1 --branch v3.7.2 https://github.com/StanfordVL/BEHAVIOR-1K.git $BH/B1K
# tag v3.7.2 = commit 88454bd04f75dc57c00ab1f1a00bcde1ff505950 (2025-12-10)

conda create -p $BH/envs/ac-behavior python=3.10 -c conda-forge -y
ln -sfn $BH/envs/ac-behavior ~/miniforge3/envs/ac-behavior
conda activate ac-behavior

export TMPDIR=$BH/tmp PIP_CACHE_DIR=$BH/tmp/pip-cache      # keep pip off the small disk
export OMNIGIBSON_DATA_PATH=$BH/datasets && mkdir -p $OMNIGIBSON_DATA_PATH
unset EXP_PATH CARB_APP_PATH ISAAC_PATH                    # setup.sh L253-257 aborts if these are set

cd $BH/B1K
./setup.sh --omnigibson --bddl --accept-nvidia-eula --confirm-no-conda   # ~20 min, downloads ~11 GB of wheels
./setup.sh --joylo --confirm-no-conda                                    # the challenge's robot config lives here
```

`--accept-nvidia-eula` is the flag `setup.sh` offers for the Isaac Sim EULA
(it exports `OMNI_KIT_ACCEPT_EULA=YES`, L296-301; without it the script exits).
`--confirm-no-conda` only skips the "you are not in a conda env" prompt.

`--joylo` is **not optional** for this port even though it is a teleoperation
package: the challenge's robot configuration is
`joylo/gello/robots/sim_robot/og_teleop_cfg.py` (`R1_CONTROLLER_CONFIG`,
`ROBOT_RESET_JOINT_POS`) and its task table is
`joylo/sampled_task/available_tasks.yaml`. `omnigibson/learning/eval.py`
imports from it too (L15-21), which is why upstream's `--eval` requires
`--joylo`.

## 3. Two pins `setup.sh` only applies with `--new-env`

`setup.sh` installs numpy, setuptools and torch **inside the `--new-env`
branch** (L216-233). Running it against an environment you made yourself skips
them, and the Isaac Sim wheels then pull their own torch. Restore both, in this
order, after the install:

```bash
pip install "numpy<2"                 # omnigibson and isaacsim-core both require numpy<2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Skipping this fails in two ways, both seen here:

* the Isaac wheels install **torch 2.14 + cu13**, which this driver refuses —
  `CUDA initialization: The NVIDIA driver on your system is too old (found
  version 12080)` — so torch runs on the CPU;
* OmniGibson's `omnigibson/utils/transform_utils.py` is `torch.compile`d, and
  torch 2.14's inductor emits `-std=c++20`, which gcc 9 on Ubuntu 20.04 does
  not know: `g++: error: unrecognized command line option '-std=c++20'`.
  torch 2.6 does not compile that path.

`pip install -e joylo` also upgrades numpy to 2.x (its `opencv-contrib-python`
asks for it); re-run `pip install "numpy<2"` after it. The resulting
"opencv-contrib-python requires numpy>=2" warning is harmless — OmniGibson
does not use that build of OpenCV.

## 4. Data — ~36 GB, and a licence you must accept

Three bundles, all under `$OMNIGIBSON_DATA_PATH` (OmniGibson's own root;
`omnigibson/macros.py` `determine_data_path` L110-128 reads that variable and
**asserts the directory already exists**). This is *not*
`EMBODIEDSCORE_DATA_ROOT`: OmniGibson resolves its assets by its own root, and
the package's loader must be given the same one.

```bash
export OMNIGIBSON_DATA_PATH=$BH/datasets
cd $BH/B1K && ./setup.sh --omnigibson --bddl --dataset \
    --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
```

| bundle | download | on disk | what it is |
|---|---|---|---|
| `omnigibson-robot-assets` | 0.47 GiB | 2.4 GB | the robot USDs, R1 Pro included |
| `behavior-1k-assets` (`…-3.7.2rc1.zip`) | 29.1 GiB | 33 GB | the scenes and every object model — **encrypted** |
| `2025-challenge-task-instances` | 0.10 GiB | 400 MB | the 301 sampled initial states per task, and the metadata the budgets come from |
| `omnigibson.key` | — | 44 B | the Fernet key `download_key()` fetches |

**Licence.** The BEHAVIOR Data Bundle EULA (rev. 8 December 2022) gates the
key: `download_behavior_1k_assets(accept_license=False)` loops on an
interactive `[y/n]` prompt. `setup.sh --accept-dataset-tos` is the flag
upstream offers to answer it non-interactively (it passes
`accept_license=True`, L418). The terms are worth reading before you use it:
the data may be used for **non-commercial academic research only**, may not be
extracted from OmniGibson or reverse-engineered, and neither the key nor the
data may be redistributed.

If you would rather not run `setup.sh` a third time (it re-checks the whole
install), the three downloads are three one-liners — this is exactly what
`setup.sh --dataset` runs (L411-425):

```bash
python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets as f; f()"
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets as f; f(accept_license=True)"
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances as f; f()"
```

The layout you end up with:

```
$OMNIGIBSON_DATA_PATH/
  omnigibson.key
  omnigibson-robot-assets/
  behavior-1k-assets/
  2025-challenge-task-instances/
    metadata/{test_instances.csv, episodes.jsonl, B50_task_misc.csv, B50_object_instance_ID.csv}
    scenes/<scene model>/json/<scene model>_task_<task>_instances/*-tro_state.json
```

The package's loader reads exactly two of those files —
`metadata/test_instances.csv` (which 10 instances of each task the public
leaderboard scores) and `metadata/episodes.jsonl` (the 200 human demos per
task, whose mean length is the task's step budget). Everything else is read by
OmniGibson itself.

The 2025 demonstration corpus (`behavior-1k/2025-challenge-demos`, ~2 TB) is
**not** needed: evaluation reads none of it.

## 5. The package

```bash
pip install -e /data/ws_vln/coding-agents-wt/behavior/thirdparty/EmbodiedScore-envs
pytest tests/test_behavior_contracts.py        # declarations + the loader; no simulator
```

The declaration tier runs anywhere. The loader tier needs the
`2025-challenge-task-instances` metadata under `OMNIGIBSON_DATA_PATH`; the BDDL
tier needs the `bddl` package, which is pip-installable on its own
(`pip install -e $BH/B1K/bddl3`, or `pip install bddl` for 3.6.0 from PyPI —
the monorepo's 3.7.0 was never published).

## 6. Verify

```bash
export OMNIGIBSON_DATA_PATH=$BH/datasets OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_APPDATA_PATH=$BH/tmp/og_appdata TMPDIR=$BH/tmp

# 1. Isaac Sim 4.5 boots and a house loads (the slow one; see the timings below)
python scripts/behavior_scripted_episode.py --task turning_on_radio --check

# 2. the full stack: reset / the three macro moves / the metrics / gymnasium's checker
EMBODIEDSCORE_RUN_BEHAVIOR_TESTS=1 pytest tests/test_behavior_contracts.py

# 3. one episode by hand
python - <<'EOF'
import embodiedscore_envs as es
env = es.make("behavior-1k", "mini")
obs, info = env.reset(options={"episode": 0})
print(info["episode"]["instruction"], "|", info["language"])
print(obs["rgb"].shape, info["n_predicates"], "goal predicates")
print(info["metrics"])
env.close()
EOF
```

Timings seen here (RTX 3090, cold caches): `import omnigibson` ~5 s; Isaac Sim
app ready ~27 s; the first load of a house several minutes (USD parsing is
CPU-bound and single-scene); a same-scene task change is much cheaper
(`env.update_task`); a same-task instance change is instant. The challenge's
own note (`docs/challenge/evaluation.md`, RTX 4090) is 150-300 s per scene load
and 20-25 FPS at 224².

## 7. Runtime knobs

| variable | effect |
|---|---|
| `OMNIGIBSON_DATA_PATH` | the asset root; the directory must exist before `import omnigibson` |
| `OMNIGIBSON_APPDATA_PATH` | Kit's caches — put it on a fast local disk |
| `OMNIGIBSON_HEADLESS=1` | no window (the port sets it) |
| `OMNIGIBSON_GPU_ID` | which GPU |
| `OMNI_KIT_ACCEPT_EULA=YES` | required by Isaac Sim at every launch |
| `EMBODIEDSCORE_RUN_BEHAVIOR_TESTS=1` | opt in to the simulator tier of the contract tests |

## 8. When it fails

| symptom | cause | fix |
|---|---|---|
| `ERROR: Python 3.10 required, found 3.x` | `setup.sh` L248 | the environment must be exactly 3.10 |
| `ERROR: Found existing Isaac Sim environment variables` | `EXP_PATH` / `CARB_APP_PATH` / `ISAAC_PATH` set by another Isaac install | unset all three, then re-run |
| `g++: error: unrecognized command line option '-std=c++20'` | torch 2.14's inductor against gcc 9 | § 3: pin torch 2.6.0+cu124 |
| `CUDA initialization: The NVIDIA driver on your system is too old` | the cu13 torch the Isaac wheels pull | § 3 |
| `AssertionError: Data path … does not exist!` on import | `OMNIGIBSON_DATA_PATH` points nowhere | `mkdir -p` it first (`macros.py` L110-128 asserts) |
| `ModuleNotFoundError: No module named 'gello'` | JoyLo not installed | § 2, `--joylo` |
| `assert isaac_version_tuple in m.KIT_FILES` | Isaac Sim 5.1 in this interpreter | v3.7.2 accepts 4.5.0 only; use a separate environment |
| `FileNotFoundError: … -tro_state.json` from `reset()` | the challenge instances are missing | § 4 |
| `Only one Simulator instance can be created at a time!` | a second `og.Environment` in one process | the world clears the stage (`og.clear()`) instead; never build two |
| the process exits silently at the end of a run | `og.shutdown()` calls `exit(0)` when Isaac is up (`omnigibson/__init__.py` L144-157) | the port never calls it — `close()` clears the stage only |
| `numpy 2.x is incompatible` after installing JoyLo | its `opencv-contrib-python` | `pip install "numpy<2"` again |
