#!/usr/bin/env bash
# Launch the IsaacLab-Arena env socket server (py3.12) inside the cluster-parity
# Isaac Sim container, on the host network so a host-side RLinf py3.11 client can
# reach it over 127.0.0.1:<port>.
#
# This is the local stand-in for posttrain's separate TCP Arena sim servers. The
# server file lives in the RLinf repo (rlinf/envs/isaaclab_arena/arena_server.py) but
# runs under the Arena .venv; we mount BOTH the RLinf repo and the Arena checkout.
#
# Usage:
#   ARENA_PORT=5557 ARENA_ENV=LMDrillLiftRlD1 ARENA_NUM_ENVS=1 ./run_arena_server.sh
#
# Readiness: the server writes the bound port to $READY_FILE (a path on a mounted
# dir, visible to the host) once the env is built and it is listening.

set -euo pipefail

IMAGE="${ARENA_DOCKER_IMAGE:-nvcr.io/nvidian/gear-n2-eval:py312-2026.05.07}"
ARENA_PATH="${ISAACLAB_ARENA_PATH:-/home/juekunl/Work/IsaacLab-Arena}"
RLINF_PATH="${RLINF_PATH:-/home/juekunl/Work/RLinf}"
BRINGUP_PATH="${ARENA_BRINGUP_PATH:-/home/juekunl/Work/arena_local_bringup}"
GPU_DEV="${ARENA_GPU:-0}"
CACHE_ROOT="${ISAAC_CACHE_ROOT:-$HOME/docker/isaac-sim}"

ARENA_PORT="${ARENA_PORT:-5557}"
ARENA_ENV="${ARENA_ENV:-LMDrillLiftRlD1}"
ARENA_NUM_ENVS="${ARENA_NUM_ENVS:-1}"
READY_FILE="${READY_FILE:-$BRINGUP_PATH/arena_server.ready}"

mkdir -p "$CACHE_ROOT"/cache/{kit,ov,glcache,computecache} "$CACHE_ROOT"/{logs,data}
rm -f "$READY_FILE"

SERVER_PY="$RLINF_PATH/rlinf/envs/isaaclab_arena/arena_server.py"
INNER_CMD="python -u '$SERVER_PY' \
  --host 127.0.0.1 --port ${ARENA_PORT} --ready-file '${READY_FILE}' \
  --env-name '${ARENA_ENV}' --num-envs ${ARENA_NUM_ENVS} --device cuda:0"

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then TTY_FLAGS=(-it); fi

exec docker run --rm "${TTY_FLAGS[@]}" --gpus "\"device=${GPU_DEV}\"" --entrypoint /bin/bash \
  --network host \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${GPU_DEV}" \
  -e ISAACLAB_ARENA_PATH="$ARENA_PATH" \
  -v "$ARENA_PATH:$ARENA_PATH:rw" \
  -v "$RLINF_PATH:$RLINF_PATH:rw" \
  -v "$BRINGUP_PATH:$BRINGUP_PATH:rw" \
  -v "$CACHE_ROOT/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov:rw" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_ROOT/cache/computecache:/root/.nv/ComputeCache:rw" \
  "$IMAGE" -lc "cd '$ARENA_PATH' && source .venv/bin/activate && ${INNER_CMD}"
