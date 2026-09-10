# Installing the LIBERO lines

Three benchmarks share the `libero` engine and each ships its own `libero`
package, so each needs an interpreter of its own:

| lines | benchmark | env | section |
|---|---|---|---|
| `libero-*` | LIBERO | `ac-libero` | §§ 1–4 below |
| `libero-pro-*` | LIBERO-PRO (arXiv:2510.03827) | `ac-libero-pro` | § LIBERO-PRO |
| `libero-plus-*` | LIBERO-Plus (arXiv:2510.13626) | `ac-libero-plus` | § LIBERO-Plus |

Do §§ 1–2 first: the other two clone `ac-libero` and replace its `libero`.

The `libero-*` lines run LIBERO (Liu et al. 2023, arXiv:2306.03310) in process:
robosuite 1.4.1 on MuJoCo, rendered headless through EGL. Nothing of it is a pip
dependency of this package — like habitat-sim, it is installed once into the
interpreter that serves these lines. The package imports it lazily: `import
embodiedscore_envs` needs none of it; a `libero-*` line asks for it when built.

## 1. The environment

One conda env for the LIBERO lines (the habitat build and robosuite pin
different numpy / torch stacks; keep them apart):

```
mamba create -y -n ac-libero python=3.10
~/miniforge3/envs/ac-libero/bin/pip install 'numpy<2' 'robosuite==1.4.1' mujoco bddl \
    'torch' scipy transforms3d imageio opencv-python-headless pyyaml easydict hydra-core cloudpickle einops
```

## 2. LIBERO, pinned, with two patches

```
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && git checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c   # 2023-12-24 "fixed the venv issue"
```

Two lines need changing before `pip install -e .` (the upstream master of
2025-03, 8f1084e, still lacks both):

| file | change | why |
|---|---|---|
| `libero/libero/benchmark/__init__.py` | `torch.load(init_states_path)` → `torch.load(init_states_path, weights_only=False)` | the init states are torch pickles; torch ≥ 2.6 refuses them by default |
| `libero/libero/envs/venv.py` | `import gym` → `import gymnasium as gym` | `gym` is not installed; only its name is used |

```
~/miniforge3/envs/ac-libero/bin/pip install -e .
echo N | ~/miniforge3/envs/ac-libero/bin/python -c "import libero"   # first import asks about custom paths; N writes ~/.libero/config.yaml
```

`~/.libero/config.yaml` records where the BDDL files, init states and assets
live (inside the checkout); the loader reads them through
`libero.libero.get_libero_path`. No `EMBODIEDSCORE_DATA_ROOT` entry, no
datasets: evaluation needs only the BDDL files and init states the checkout
ships.

## 3. This package, and a check

```
~/miniforge3/envs/ac-libero/bin/pip install -e /path/to/EmbodiedScore-envs
MUJOCO_GL=egl ~/miniforge3/envs/ac-libero/bin/python -c "
import embodiedscore_envs as es
env = es.make('libero-spatial', 'mini'); obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, info['episode']['instruction'], info['metrics']); env.close()"
```

`MUJOCO_GL=egl` is set by `LiberoWorld` when unset; an NVIDIA driver is needed
for EGL. The first `reset` of a task builds its MuJoCo model (~5 s); later
episodes of the same task reuse it.

## 4. Tests

```
MUJOCO_GL=egl EMBODIEDSCORE_RUN_LIBERO_TESTS=1 ~/miniforge3/envs/ac-libero/bin/python -m pytest tests/test_libero_contracts.py
```

Without the variable the file runs only the declaration contracts (no simulator).

---

# LIBERO-PRO

`libero-pro-*` (`benchmarks/libero_pro.py`). LIBERO-PRO is a **fork of LIBERO**
that installs under the same import name, so it cannot share `ac-libero`: it
gets a clone of it. Nothing of `ac-libero` is touched.

```
conda create -y --clone ac-libero -n ac-libero-pro
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git /data/ws_vln/coding-agents/data/libero_pro/repo
cd /data/ws_vln/coding-agents/data/libero_pro/repo && git checkout eafdb809426b13153aa1e4c42d6601844217dfec
```

Three edits before installing — the two of § 2 above (the fork is the same
2023 code), plus one the release is missing:

| file | change | why |
|---|---|---|
| `libero/libero/benchmark/__init__.py` | `torch.load(init_states_path)` → `torch.load(init_states_path, weights_only=False)` | torch ≥ 2.6 refuses the init-state pickles |
| `libero/libero/envs/venv.py` | line 3, `import gym` → `import gymnasium as gym` | `gym` is not installed; only its name is used |
| `libero/__init__.py` | `touch` it | upstream LIBERO tracks it as an empty file; the fork was uploaded through the GitHub web form, which drops empty files, so `find_packages()` misses the `libero` package and `import libero` fails |

```
~/miniforge3/envs/ac-libero-pro/bin/pip uninstall -y libero
~/miniforge3/envs/ac-libero-pro/bin/pip install -e /data/ws_vln/coding-agents/data/libero_pro/repo
~/miniforge3/envs/ac-libero-pro/bin/pip install -e /path/to/EmbodiedScore-envs
```

**The task files.** The perturbed BDDL files and init states are not in the
repo — `perturbation.py` generates them, and the generated release is the
HuggingFace dataset `zhouxueyang/LIBERO-Pro` (676 files, 15 MB). Download it and
unpack the four released dimensions into the fork, where its own benchmark
registry expects them (`libero_<suite>_{object,swap,lan,task}`):

```
hf download zhouxueyang/LIBERO-Pro --repo-type dataset \
    --local-dir /data/ws_vln/coding-agents/data/libero_pro/hf_tasks
SRC=/data/ws_vln/coding-agents/data/libero_pro/hf_tasks
DST=/data/ws_vln/coding-agents/data/libero_pro/repo/libero/libero
for dim in lan object swap task; do for s in libero_spatial libero_object libero_goal libero_10; do
    cp -r "$SRC/bddl_files/${s}_${dim}" "$DST/bddl_files/"
    cp -r "$SRC/init_files/${s}_${dim}" "$DST/init_files/"
done; done
```

**The config file.** `libero.libero.get_libero_path` reads
`$LIBERO_CONFIG_PATH/config.yaml`, default `~/.libero` — which belongs to
`ac-libero` and points at the ORIGINAL checkout. The fork must not read it (its
`_object` dimension needs the fork's own object classes and assets), so give
this interpreter its own, with a `.pth` (site-packages runs `import` lines in
`.pth` files at startup, before anything else, and the runner launches
`env.python` directly, never through `conda activate`):

```
echo "import os; os.environ.setdefault('LIBERO_CONFIG_PATH', '/data/ws_vln/coding-agents/data/libero_pro/libero_config')" \
    > ~/miniforge3/envs/ac-libero-pro/lib/python3.10/site-packages/libero_config_path.pth
mkdir -p /data/ws_vln/coding-agents/data/libero_pro/libero_config
PRO=/data/ws_vln/coding-agents/data/libero_pro/repo/libero/libero
cat > /data/ws_vln/coding-agents/data/libero_pro/libero_config/config.yaml <<EOF
assets: $PRO/assets
bddl_files: $PRO/bddl_files
benchmark_root: $PRO
datasets: $PRO/../datasets
init_states: $PRO/init_files
EOF
```

Disk: the checkout is 914 MB (644 MB of it vendored LIBERO assets), the task
release 15 MB. No datasets — evaluation reads neither.

```
MUJOCO_GL=egl ~/miniforge3/envs/ac-libero-pro/bin/python -c "
import embodiedscore_envs as es
env = es.make('libero-pro-spatial', 'mini'); obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, info['episode']['episode_id'], info['episode']['instruction']); env.close()"
MUJOCO_GL=egl ~/miniforge3/envs/ac-libero-pro/bin/python scripts/libero_scripted_episode.py libero-pro-spatial
MUJOCO_GL=egl EMBODIEDSCORE_RUN_LIBERO_PRO_TESTS=1 ~/miniforge3/envs/ac-libero-pro/bin/python -m pytest tests/test_libero_pro_contracts.py
```

The fifth dimension the paper names, `environment` (`_env`), ships no task
files in the repo or the release and is not a line — see the module docstring.

---

# LIBERO-Plus

`libero-plus-*` (`benchmarks/libero_plus.py`). LIBERO-Plus is a **drop-in
replacement** for the `libero` package (not a GitHub fork), so it too gets its
own clone of `ac-libero`.

```
conda create -y --clone ac-libero -n ac-libero-plus
git clone https://github.com/sylvestf/LIBERO-plus.git /data/ws_vln/coding-agents/data/libero_plus/repo
cd /data/ws_vln/coding-agents/data/libero_plus/repo && git checkout 4976dc30028e805ff8094b55501d532c48fec182
```

The same three edits as LIBERO-PRO (`torch.load` in **two** places here,
`venv.py` line 3, and the missing empty `libero/__init__.py`), then:

```
~/miniforge3/envs/ac-libero-plus/bin/pip uninstall -y libero
~/miniforge3/envs/ac-libero-plus/bin/pip install -e /data/ws_vln/coding-agents/data/libero_plus/repo
~/miniforge3/envs/ac-libero-plus/bin/pip install -e /path/to/EmbodiedScore-envs
~/miniforge3/envs/ac-libero-plus/bin/pip install 'usd-core>=25.5' wand scikit-image   # extra_requirements.txt + requirements.txt
```

`wand` and `scikit-image` are what the sensor-noise corruptions run on
(`env_wrapper.py`: motion / gaussian / zoom / glass blur, fog).

**The assets.** The BDDL files (16 330) and init states (4 170) are in the
repo; the meshes, scenes and textures are not — a 6.4 GB `assets.zip` on
HuggingFace that unpacks to 9.5 GB under a long internal prefix:

```
hf download Sylvest/LIBERO-plus --repo-type dataset \
    --local-dir /data/ws_vln/coding-agents/data/libero_plus/hf_assets
cd /data/ws_vln/coding-agents/data/libero_plus/repo/libero/libero
unzip -q /data/ws_vln/coding-agents/data/libero_plus/hf_assets/assets.zip -d .
mv inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets ./assets
rm -rf inspire
```

**The config file**, for the same reason as LIBERO-PRO:

```
echo "import os; os.environ.setdefault('LIBERO_CONFIG_PATH', '/data/ws_vln/coding-agents/data/libero_plus/libero_config')" \
    > ~/miniforge3/envs/ac-libero-plus/lib/python3.10/site-packages/libero_config_path.pth
mkdir -p /data/ws_vln/coding-agents/data/libero_plus/libero_config
PLUS=/data/ws_vln/coding-agents/data/libero_plus/repo/libero/libero
cat > /data/ws_vln/coding-agents/data/libero_plus/libero_config/config.yaml <<EOF
assets: $PLUS/assets
bddl_files: $PLUS/bddl_files
benchmark_root: $PLUS
datasets: $PLUS/../datasets
init_states: $PLUS/init_files
EOF
```

Disk: 9.7 GB in place (9.5 GB of it assets) plus the 6.0 GB zip, which can be
deleted afterwards. The training and LeRobot releases on HuggingFace (75–100 GB)
are not needed — evaluation reads none of them.

```
MUJOCO_GL=egl ~/miniforge3/envs/ac-libero-plus/bin/python -c "
import embodiedscore_envs as es
env = es.make('libero-plus-camera', 'mini'); obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, info['episode']['episode_id'], info['episode']['instruction']); env.close()"
MUJOCO_GL=egl ~/miniforge3/envs/ac-libero-plus/bin/python scripts/libero_scripted_episode.py libero-plus-camera
MUJOCO_GL=egl EMBODIEDSCORE_RUN_LIBERO_PLUS_TESTS=1 ~/miniforge3/envs/ac-libero-plus/bin/python -m pytest tests/test_libero_plus_contracts.py
```

The last test loads all seven lines on `all` and checks the released task
counts (10 030 in total); it takes about half a minute.
