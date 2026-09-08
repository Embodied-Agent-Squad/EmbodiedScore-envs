# Isaac Sim for the VLNverse lines

Only the `vlnverse-*` lines need this; the habitat lines need EmbodiedScore-habitat
(README "Install"). End to end: Isaac Sim 5.1, the render worker's dependencies,
the data, and the checks that tell you each layer works. Everything below was run
on Ubuntu 20.04 with an RTX 3090 and driver 570.133.07 through the container
route; the pip and workstation routes are NVIDIA's own instructions, quoted.

You end up with two Python interpreters: **yours** (`embodiedscore_envs`,
gymnasium, your agent) and **Isaac's** (the render worker, spawned by the
environment through the launcher named in `EMBODIEDSCORE_ISAAC_PYTHON`). With
the pip route they can be the same interpreter.

## 1. Requirements

| | NVIDIA's requirement for Isaac Sim 5.1.0 | what this package was run on |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | Ubuntu 20.04 host + the 5.1.0 container |
| driver | 580.65.06 (tested by NVIDIA); latest production branch recommended | 570.133.07 worked (observation, not a guarantee) |
| GPU | RTX with RT cores, ≥ 16 GB VRAM; A100 / H100 are not supported | RTX 3090, ~4.6 GB in use with one scene loaded |
| RAM / disk | 32 GB / 50 GB SSD for Isaac itself | + ~1 GB per scene (all 262: ~291 GB) |
| Python (pip route) | 3.11, glibc ≥ 2.35 | — |

## 2. Isaac Sim 5.1 — pick one route

### A. pip (Ubuntu 22.04+, Python 3.11)

```bash
conda create -n isaac python=3.11 && conda activate isaac
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES
export EMBODIEDSCORE_ISAAC_PYTHON="$(which python)"     # the same interpreter can host embodiedscore_envs
```

### B. Workstation bundle

Download `isaac-sim-standalone-5.1.0-linux-x86_64.zip` from NVIDIA's Isaac Sim
release page, then:

```bash
mkdir -p ~/isaacsim && unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -d ~/isaacsim
cd ~/isaacsim && ./post_install.sh
export EMBODIEDSCORE_ISAAC_PYTHON=~/isaacsim/python.sh
```

`python.sh` at the bundle root is the launcher; `~/isaacsim5.1/python.sh` is
the package's default, so a symlink there saves the variable.

### C. Container (what this package was validated on)

Prerequisites: docker and the NVIDIA Container Toolkit (`nvidia-ctk runtime
configure --runtime=docker`, then restart docker; under rootless docker also
`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). Then build the
derived image — Isaac Sim plus the worker's two dependencies, entrypoint
cleared — and use the wrapper as the launcher:

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.1.0                                    # 15 GB
docker build -t embodiedscore-isaac:5.1.0 -f scripts/Dockerfile.isaac scripts
export EMBODIEDSCORE_ISAAC_PYTHON=$PWD/scripts/isaac_container.sh
export ISAAC_GPU_FLAGS="--device nvidia.com/gpu=all"                          # rootless docker only; default is --gpus all
```

`scripts/isaac_container.sh` runs `/isaac-sim/python.sh` in the image with
stdin/stdout passed through and mounts this repository, `EMBODIEDSCORE_DATA_ROOT`
and `EMBODIEDSCORE_SCENE_ROOT` at their host paths, so set those variables (step 5)
before the first launch. It runs as container root (`-u 0:0`): under rootless
docker that is you; under rootful docker the caches and any debug output come
out root-owned. Kit caches live under `~/.cache/embodiedscore-isaac`
(`ISAAC_CACHE`).

## 3. The worker's dependencies

Routes B and C need `msgpack` and `msgpack-numpy` inside Isaac's python (the
derived image already has them; route A installs them in step 4):

```bash
$EMBODIEDSCORE_ISAAC_PYTHON -m pip install msgpack msgpack-numpy
$EMBODIEDSCORE_ISAAC_PYTHON embodiedscore_envs/benchmarks/env/sim/isaac/worker.py --check   # prints the isaacsim version, boots nothing
```

## 4. The package

`pip install -e ".[data]"` as in README "Install" (habitat-sim is not needed
for the Isaac lines — `import embodiedscore_envs` works without it), then

```bash
pytest tests/test_vlnverse_freemap.py tests/test_vlnverse_metrics.py tests/test_isaac_resample.py   # no data, no simulator
```

## 5. Data

The VLNverse corpus lives beside the habitat ones: splits under
`$EMBODIEDSCORE_DATA_ROOT/vlnverse/`, scenes under `$EMBODIEDSCORE_SCENE_ROOT/vlnverse/`.

```bash
python scripts/download_vlnverse_data.py splits                                         # Eyz/VLNVerse_data, raw_data/final_splits/ only (9 MB)
python scripts/download_vlnverse_data.py scenes --split fine/val_unseen --limit 5       # Eyz/VLNVerse_scene, ~1 GB per scene
python scripts/download_vlnverse_data.py check --split fine/val_unseen                  # what the split still lacks
```

`scenes --split` fetches the scenes a split uses in episode order (fine/val_unseen: 33 scenes, ~38 GB; `--limit 5` covers its first 100 episodes); `--scan kujiale_0272 ...` names scenes directly. If Hugging Face refuses the download, `huggingface-cli login` first.

`freemap.npy` — the occupancy grid the agent moves on, one per scene, 130 MB
for all — is not in the Hugging Face scene release at the time of writing; the
environment raises at the first move of a scene without one. `check` lists
the scenes that lack it.

```
$EMBODIEDSCORE_DATA_ROOT/vlnverse/raw_data/final_splits/{fine,coarse}_{train,val,val_unseen,test}.json.gz
$EMBODIEDSCORE_SCENE_ROOT/vlnverse/<scan>/start_result_navigation.usd   + Meshes/  Materials/
$EMBODIEDSCORE_SCENE_ROOT/vlnverse/<scan>/freemap.npy
```

## 6. Verify

```bash
# 1. Isaac renders a scene (boots Isaac, ~1 min; writes debug_outputs/sim_worker_selftest/view0.png)
$EMBODIEDSCORE_ISAAC_PYTHON embodiedscore_envs/benchmarks/env/sim/isaac/worker.py --selftest --scene-id kujiale_0272

# 2. the full stack: reset / step / STOP / panorama / seeding / gymnasium's checker / the upstream variant (~5 min)
EMBODIEDSCORE_RUN_ISAAC_TESTS=1 pytest tests/test_vlnverse_contracts.py

# 3. one episode by hand
python - <<'EOF'
import embodiedscore_envs as es
env = es.make("vlnverse-fine", "val_unseen")
obs, info = env.reset(options={"episode": 0})
print(info["episode"]["instruction"]); print(obs["rgb"].shape, obs["depth"].shape)
for a in (es.Act.FORWARD, es.Act.LEFT, es.Act.STOP):
    obs, r, term, trunc, info = env.step(a)
print(info["metrics"]); env.close()
EOF
```

Timings seen here: Isaac boot ~20 s; first load of a scene 30 s to 3 min
depending on its size and whether Kit's caches are warm; a same-scene episode
change is instant; one frame ~1 s; a 12-view panorama ~8 s.

## 7. Runtime knobs

| variable | effect |
|---|---|
| `EMBODIEDSCORE_ISAAC_PYTHON` | Isaac's python launcher (default `~/isaacsim5.1/python.sh`) |
| `EMBODIEDSCORE_ISAAC_SOCKET` | attach to a worker you started yourself: `$EMBODIEDSCORE_ISAAC_PYTHON embodiedscore_envs/benchmarks/env/sim/isaac/worker.py --listen /tmp/isaac.sock` — one Isaac boot for many short runs |
| `EMBODIEDSCORE_ISAAC_HEADLESS=0` | open the Isaac window (routes A and B) |
| `EMBODIEDSCORE_ISAAC_RENDERER` | `RaytracedLighting` (default) or `PathTracing` |
| `ISAAC_IMAGE` · `ISAAC_CACHE` · `ISAAC_GPU_FLAGS` | container wrapper (route C) |

## 8. When it fails

| symptom | cause | fix |
|---|---|---|
| `sim worker closed the pipe before the frame-channel magic; launcher output was: 'There was an error running python'` | Isaac's python could not open the worker script | route C: the repo is not mounted — run the wrapper from this repository, or set `EMBODIEDSCORE_ISAAC_PYTHON` to its absolute path |
| `ModuleNotFoundError: No module named 'msgpack'` in the worker's stderr | worker deps missing in Isaac's python | step 3 |
| `omni.datastore` lock / `shaderdb` errors, then a `TypeError` from `get_depth()` and a segfault | the container runs as uid 1234 and cannot write your mounts | keep `-u 0:0` (the wrapper's default) or `chown 1234` the cache and output directories |
| the container starts the full headless Kit app instead of the worker | the base image's ENTRYPOINT is `runheadless.sh` | use the derived image (`ENTRYPOINT []`) or pass `--entrypoint /isaac-sim/python.sh` |
| `ERROR_OUT_OF_DEVICE_MEMORY` during scene load | another process holds the GPU | free it; one scene needs ~4.6 GB, the load peak is higher |
| `RuntimeError: use_occupancy_collision=True but no freemap loaded` | `freemap.npy` missing for that scene | step 5 |
| `Failed to upload DomeLight texture ./limpopo_golf_course_4k.hdr` in the log | the released USD stages reference an HDR that is not shipped | harmless; the frame renders without it |
| a `Warning: running in conda env` line from `python.sh` | Isaac's launcher prints to stdout before python starts | harmless; the client skips everything before the frame-channel preamble |
