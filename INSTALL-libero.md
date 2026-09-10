# Installing the LIBERO lines

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
