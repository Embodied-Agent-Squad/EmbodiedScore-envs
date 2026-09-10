# Installing the CALVIN line

The `calvin-d` line runs CALVIN in process: `calvin_env` on **pybullet**,
rendered headless through pybullet's EGL plugin (no display, no MuJoCo, no
torch). Nothing of it is a pip dependency of this package — like habitat-sim,
LIBERO, RoboTwin and RoboCasa, it is installed once into the interpreter that
serves the line. The package imports it lazily: `import embodiedscore_envs`
needs none of it, and even the episode list loads without it (an episode is a
symbolic initial condition and five task names); the first `reset()` is where
the simulator is required.

| | |
|---|---|
| paper | Mees et al., RA-L 2022, [arXiv:2112.03227](https://arxiv.org/abs/2112.03227) |
| repo | [github.com/mees/calvin](https://github.com/mees/calvin), `main` at commit **`fa03f01`** |
| submodule | [`calvin_env`](https://github.com/mees/calvin_env) at **`1431a46`** |
| robot | Franka Panda (`panda_longer_finger.urdf`), parallel gripper, 30 Hz over 240 Hz pybullet |
| environments | A / B / C / D — the board evaluates in **D** (both ABC→D and ABCD→D) |
| tasks | 34 language-conditioned |
| evaluation | 1000 chains of 5 instructions, 360 ticks per instruction |
| env here | `ac-calvin` (python 3.10) |
| data | `calvin_debug_dataset`, 1.3 GB (only one 7 KB file is read) |

## 1. The clone

```
mkdir -p /data/ws_vln/coding-agents/data/calvin
cd /data/ws_vln/coding-agents/data/calvin
git clone --recurse-submodules https://github.com/mees/calvin.git calvin-src
cd calvin-src && git checkout fa03f01 && git submodule update --init --recursive
```

`git submodule status` must show `1431a46 calvin_env`.

## 2. The environment

**Do not run the repo's `install.sh`.** It installs `calvin_env/tacto` (a
DIGIT tactile renderer) and `calvin_models` — the latter pins an old torch and
pytorch-lightning for CALVIN's own policies, which this package never runs. The
*environment* needs none of that:

```
mamba create -y -n ac-calvin python=3.10
conda activate ac-calvin
pip install --no-cache-dir "numpy==1.26.4" "pybullet==3.2.6" "hydra-core==1.1.1" \
    "hydra-colorlog==1.1.0" "omegaconf==2.1.1" "gym==0.26.2" scipy opencv-python-headless \
    numpy-quaternion pandas matplotlib cloudpickle gitpython rich \
    "gymnasium>=1.3" fastdtw pillow "msgpack>=1.0" msgpack-numpy pytest

pip install --no-deps -e /data/ws_vln/coding-agents/data/calvin/calvin-src/calvin_env
pip install --no-deps -e /path/to/EmbodiedScore-envs
```

`--no-deps` on `calvin_env` keeps its unpinned `requirements.txt` from moving
numpy; every runtime import it makes is in the list above (`gym` is the old
OpenAI gym — `PlayTableSimEnv` subclasses `gym.Env` — and coexists with
`gymnasium`).

`tacto` is **not** installed and not needed: the dataset's config declares a
third camera, `tactile`, which the world drops through `get_env`'s own
`obs_space` filter (`sim/calvin/world.py`, `OBS_SPACE`).

`pyhash` is **not** installed either. The evaluator uses it to derive a
per-condition random seed; it does not build on python ≥ 3.10, so
`benchmarks/calvin.py` transcribes the 20 lines of it that matter (`fnv1_32`,
with pyhash's two surprises: a UTF-16 encoding of the string and a seed of 0
rather than the FNV-1 offset basis).

## 3. The data

Only the **environment config** is read, and every released dataset carries the
same one. Take the smallest:

```
cd /data/ws_vln/coding-agents/data/calvin
wget http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip   # 1.3 GB
unzip calvin_debug_dataset.zip && rm calvin_debug_dataset.zip
```

That is `dataset/download_data.sh debug`. What the line actually opens is one
file:

```
calvin_debug_dataset/validation/.hydra/merged_config.yaml     7.3 KB
```

which is what `evaluate_policy.make_env` hands `get_env`
(`calvin_models/calvin_agent/evaluation/evaluate_policy.py`:48 →
`calvin_env/envs/play_table_env.py`:275). It names `calvin_scene_D`, the
`panda_longer_finger` robot, and the two cameras. The 1681 `episode_*.npz`
play-data frames beside it are training data and are never read — the
evaluator's initial states are *computed*, not loaded (see `benchmarks/
calvin.py`). `task_D_D` (166 GB), `task_ABC_D` (517 GB) and `task_ABCD_D`
(656 GB) carry the identical config; there is no reason to download one.

The play-table meshes and URDFs live inside the `calvin_env` checkout
(`calvin_env/data/`), resolved by `PlayTableScene` relative to the package.

Point the package at it:

```
export EMBODIEDSCORE_DATA_ROOT=/data/ws_vln/coding-agents/data
```

so the line resolves `$EMBODIEDSCORE_DATA_ROOT/calvin/calvin_debug_dataset/`.
A different folder name goes through the loader kwarg `calvin_dataset`.

## 4. The evaluation sequences

CALVIN's 1000 chains are **generated, not stored**: `get_sequences(1000)`
(`multistep_sequences.py`:350) plans them from 192 symbolic initial conditions
under fixed seeds. `benchmarks/calvin.py` transcribes that generator (upstream
farms the per-condition draws to a process pool; the transcription is serial
with the same per-condition seeding, and produces the identical 1000 pairs).
It takes **about two minutes** the first time and is then cached at

```
$EMBODIEDSCORE_DATA_ROOT/calvin/eval_sequences_1000.json     ~180 KB
```

Delete that file to regenerate.

## 5. A check

```
export EMBODIEDSCORE_DATA_ROOT=/data/ws_vln/coding-agents/data
python -c "
import embodiedscore_envs as es
env = es.make('calvin-d', 'mini')
obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, obs['wrist'].shape)
print(info['instruction'], '|', info['subtask'], '| chain', info['metrics'])
env.close()"
```

pybullet prints `Loading EGL plugin` and the GL renderer on the first build;
an NVIDIA driver is needed for EGL. Building the table costs a few seconds
once — unlike RoboCasa, every CALVIN episode reuses the same scene, so a reset
is only a state write and costs milliseconds.

## 6. Tests

```
python -m pytest tests/test_calvin_contracts.py
```

runs the declaration and table contracts with no simulator. Add the data root
to get the loader tests, and

```
EMBODIEDSCORE_DATA_ROOT=/data/ws_vln/coding-agents/data \
EMBODIEDSCORE_RUN_CALVIN_TESTS=1 python -m pytest tests/test_calvin_contracts.py
```

for the simulator ones. Two of them check this package's transcriptions against
whatever checkout sits next to the installed `calvin_env`, so a repo bump is
caught here rather than in a run: `test_instructions_match_the_checkout`
(the 34 validation annotations) and `test_sequence_tables_match_the_checkout`
(`multistep_sequences.tasks` / `task_categories`, which decide which chains
exist and in what order they are drawn).
`test_task_oracle_table_matches_the_installed_calvin_env` does the same for the
success predicates. The full 1000-sequence comparison against upstream's own
generator is behind `EMBODIEDSCORE_RUN_CALVIN_SEQ_TEST=1` (minutes).

Then the end-to-end proof, which needs no model:

```
EMBODIEDSCORE_DATA_ROOT=/data/ws_vln/coding-agents/data \
python scripts/calvin_scripted_episode.py --episode 0
```

(that script lives in the coding-agents repo, not this package.)
