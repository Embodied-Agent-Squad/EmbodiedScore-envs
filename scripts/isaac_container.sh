#!/usr/bin/env bash
# Isaac Sim 5.1 launcher backed by a container — a drop-in for EMBODIEDSCORE_ISAAC_PYTHON.
#
# Runs `/isaac-sim/python.sh "$@"` inside the embodiedscore-isaac image with stdin
# and stdout passed straight through, so the render worker's msgpack frame
# channel works unchanged. This repository, $EMBODIEDSCORE_DATA_ROOT and
# $EMBODIEDSCORE_SCENE_ROOT are mounted at the SAME paths inside the container, so
# every path the environment hands the worker (worker script, USD stages,
# freemaps) resolves as-is. Kit caches persist under $ISAAC_CACHE.
#
#   ISAAC_IMAGE      image to run          (default embodiedscore-isaac:5.1.0, see Dockerfile.isaac)
#   ISAAC_CACHE      host cache directory  (default ~/.cache/embodiedscore-isaac)
#   ISAAC_GPU_FLAGS  how docker sees the GPU (default "--gpus all"; rootless docker with a
#                    CDI spec: "--device nvidia.com/gpu=all")
#
# Why -u 0:0: the image's default user is uid 1234, which cannot write mounts
# owned by you (Kit's datastore lock and shader cache fail, the camera returns
# nothing). Under rootless docker, container root IS your user. Under rootful
# docker, files the container writes (caches, debug outputs) come out root-owned.
#
# Why exec + --init: a SIGTERM from the parent (proc.terminate()) is proxied by
# the docker client into the container and forwarded by PID 1; a hard kill of
# the client closes stdin, the worker sees EOF and exits, --rm reaps the container.
set -u

IMAGE="${ISAAC_IMAGE:-embodiedscore-isaac:5.1.0}"
CACHE="${ISAAC_CACHE:-$HOME/.cache/embodiedscore-isaac}"
GPU_FLAGS="${ISAAC_GPU_FLAGS:---gpus all}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

mounts=(-v "$REPO:$REPO")
mount_real() {                              # mount a path and, when it is a symlink, its target too —
  local d="$1" real                         # a symlink into an unmounted tree resolves to NOTHING inside the
  [ -n "$d" ] && [ -e "$d" ] || return 0    # container, and Isaac then renders an empty stage (black frames,
  mounts+=(-v "$d:$d")                      # depth 20 m everywhere) instead of failing
  real="$(readlink -f "$d")"
  [ "$real" != "$d" ] && mounts+=(-v "$real:$real")
  return 0
}
for d in "${EMBODIEDSCORE_DATA_ROOT:-}" "${EMBODIEDSCORE_SCENE_ROOT:-}"; do
  mount_real "$d"
  [ -n "$d" ] && [ -L "$d/vlnverse" ] && mount_real "$d/vlnverse"   # the corpus dir of a symlink farm
done

for sub in kit ov pip glcache computecache logs data; do mkdir -p "$CACHE/$sub"; done

WORKDIR="$PWD"
case "$WORKDIR" in "$REPO"/*|"$REPO") ;; *) WORKDIR=/isaac-sim ;; esac

# shellcheck disable=SC2086
exec docker run --rm -i --init -u 0:0 \
  --name "isaac51-$$" \
  $GPU_FLAGS \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$CACHE/kit:/isaac-sim/kit/cache" \
  -v "$CACHE/ov:/root/.cache/ov" \
  -v "$CACHE/pip:/root/.cache/pip" \
  -v "$CACHE/glcache:/root/.cache/nvidia/GLCache" \
  -v "$CACHE/computecache:/root/.nv/ComputeCache" \
  -v "$CACHE/logs:/root/.nvidia-omniverse/logs" \
  -v "$CACHE/data:/root/.local/share/ov/data" \
  "${mounts[@]}" \
  -w "$WORKDIR" \
  --entrypoint /isaac-sim/python.sh \
  "$IMAGE" "$@"
