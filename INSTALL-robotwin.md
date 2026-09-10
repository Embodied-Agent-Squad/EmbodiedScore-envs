# Installing the RoboTwin lines

The `robotwin-*` lines run RoboTwin 2.0 (Chen et al. 2026,
[arXiv:2506.18088](https://arxiv.org/abs/2506.18088), ICML 2026) in process:
SAPIEN 3 with its Vulkan ray tracer, headless. Nothing of it is a pip dependency
of this package — like habitat-sim and LIBERO, it is installed once into the
interpreter that serves these lines, and RoboTwin itself is a checkout rather
than a package. The package imports it lazily: `import embodiedscore_envs` needs
none of it; a `robotwin-*` line asks for it when built.

Needs an NVIDIA GPU with a working Vulkan ICD (`vulkaninfo --summary` must list
the NVIDIA driver, not just lavapipe) and CUDA — curobo compiles kernels against
it. Verified 2026-09-10 on an RTX 3090, driver 570.133.07, CUDA 12.1.

## 1. RoboTwin, pinned

```
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin && git checkout 96c1feab536306b50c26af200044fcdf126e8904   # main, 2026-09
```

The `XPolicyLab` submodule is NOT needed: it carries the policy-serving and
training stack, and this package drives the simulator directly.

## 2. The environment

One conda env for the RoboTwin lines (SAPIEN 3 and RoboTwin's numpy 1.x /
torch 2.4 stack do not share an interpreter with the habitat build or with
robosuite):

```
conda create -y -n ac-robotwin python=3.10
~/miniforge3/envs/ac-robotwin/bin/python -m pip install -r <RoboTwin>/scripts/requirements.txt
~/miniforge3/envs/ac-robotwin/bin/python -m pip install 'setuptools==69.5.1'
```

`setuptools` must stay below 81: `sapien/__init__.py` imports `pkg_resources`,
which newer setuptools no longer ships. RoboTwin's own installer pins the same
version at the end for the same reason.

Then the two source patches RoboTwin's `scripts/_install.sh` applies — this
package does not apply them for you:

| file | change | why |
|---|---|---|
| `sapien/wrapper/urdf_loader.py` | `open(..., "r")` → `open(..., "r", encoding="utf-8")` (lines 667, 673) | the embodiment URDF/SRDF are UTF-8 |
| `mplib/planner.py` | drop `or collide` from the screw-plan guard (line 807) | the guard rejects valid plans |

```
SP=$(~/miniforge3/envs/ac-robotwin/bin/python -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SP/sapien/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$SP/mplib/planner.py"
```

## 3. curobo — RoboTwin's motion planner

`Robot.set_planner` (`envs/robot/robot.py:257`) builds a `CuroboPlanner`
unconditionally, so curobo is required, not optional. `mplib` stays installed:
RoboTwin uses it for the TOPP time-parameterisation of a joint-target action.

```
cd <RoboTwin>/envs
git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 \
  ~/miniforge3/envs/ac-robotwin/bin/python -m pip install -e . --no-build-isolation
~/miniforge3/envs/ac-robotwin/bin/python -m pip install warp-lang==1.12.0 'setuptools==69.5.1'
```

`TORCH_CUDA_ARCH_LIST` is this GPU's compute capability (8.6 = Ampere / RTX
3090); leaving it unset builds every architecture and takes far longer. The
build takes about 20 minutes. It upgrades `scipy` past RoboTwin's pin
(1.10.1 → 1.15.3); that has caused no trouble here.

## 4. Assets

RoboTwin downloads its meshes, embodiments and textures from Hugging Face
(`TianxingChen/RoboTwin2.0`) into its own `assets/`:

```
cd <RoboTwin> && bash scripts/_download_assets.sh
```

or, equivalently, `python assets/_download.py` followed by unzipping the three
archives in place and then, from the checkout root:

```
python ./scripts/update_embodiment_config_path.py
```

That last step is required: it expands `${ASSETS_PATH}` in each embodiment's
`curobo_*_tmp.yml` into the absolute `curobo_left.yml` / `curobo_right.yml` the
planner loads. Run it again whenever the checkout moves.

Sizes as downloaded (2026-09-10):

| archive | download | unpacked |
|---|---|---|
| `embodiments.zip` | 210 MiB | 901 MiB |
| `objects.zip` | 3.5 GiB | 4.4 GiB |
| `background_texture.zip` | 10.2 GiB | 11 GiB |
| total under `assets/` | 14 GiB | 16 GiB |

`background_texture/` is only read when a line randomises the background
(`robotwin-randomized`); `robotwin-clean` never touches it.

## 5. This package, and a check

```
~/miniforge3/envs/ac-robotwin/bin/python -m pip install 'gymnasium==1.3.0' -e /path/to/EmbodiedScore-envs
```

RoboTwin pins `gymnasium==0.29.1`; this package needs 1.3. RoboTwin only
subclasses `gymnasium.Env` and never uses the parts that changed, so the upgrade
is safe — it has been the running configuration here since 2026-09-10.

`EMBODIEDSCORE_ROBOTWIN_ROOT` points at the checkout (or pass `robotwin_root=`
to `es.make`):

```
EMBODIEDSCORE_ROBOTWIN_ROOT=<RoboTwin> ~/miniforge3/envs/ac-robotwin/bin/python -c "
import torch.multiprocessing as mp; mp.set_start_method('spawn', force=True)
import embodiedscore_envs as es
env = es.make('robotwin-clean', 'mini'); obs, info = env.reset(options={'episode': 0})
print(obs['rgb'].shape, info['episode']['instruction'], info['metrics']); env.close()"
```

Two things the caller owns:

* **The multiprocessing start method must be `spawn`.** RoboTwin runs one curobo
  planner per arm in a child process (`envs/robot/robot.py:294`); forking after
  CUDA is initialised breaks it. RoboTwin's own entrypoints set `spawn` too
  (`scripts/collect_data.py`).
* **`SAPIEN_HEADLESS=1`** is set by `RobotwinWorld` when unset.

The first `reset` of a task costs about 40 s (SAPIEN scene, URDF, curobo warm-up
and RoboTwin's 2500-step settle); later episodes of the same task are 3-7 s. A
macro step is well under a second.

## 6. The scripted-expert proof

RoboTwin ships a scripted expert per task. This drives one episode's expert
waypoints through the package's macro pose protocol and exits non-zero unless
`success == 1.0`:

```
EMBODIEDSCORE_ROBOTWIN_ROOT=<RoboTwin> ~/miniforge3/envs/ac-robotwin/bin/python \
  scripts/robotwin_expert_episode.py --line robotwin-clean --split mini --episode 0
```

## 7. Tests

```
EMBODIEDSCORE_ROBOTWIN_ROOT=<RoboTwin> EMBODIEDSCORE_RUN_ROBOTWIN_TESTS=1 \
  ~/miniforge3/envs/ac-robotwin/bin/python -m pytest tests/test_robotwin_contracts.py
```

Without the variable the file runs only the declaration contracts (no simulator).
