#!/usr/bin/env bash
# Run an arbitrary command inside the cluster-parity Isaac Sim container with the
# local IsaacLab-Arena checkout (and its .venv) mounted.
#
# This is the local bring-up runtime that matches the gr00t eval cluster:
#   - image  : nvcr.io/nvidian/gear-n2-eval:py312-2026.05.07  (isaac-sim 6.0.0-rc.22 / "dev2")
#   - runtime: the Arena .venv (pip isaacsim 6.0.0.0 + isaaclab 4.6.5 + isaaclab_arena)
#              activated inside the container, using the CONTAINER's GPU/Vulkan/driver
#              stack. This combination renders headless cameras correctly, unlike a
#              bare host run of pip isaacsim 6.0.0.0 (which crashes during reset()).
#
# Why this exists: a host-native headless render crashes ~30-40s into reset() with
# DLSS "below minimal input resolution of 300" + SdrShaderNode material faults and a
# silent exit. Running the SAME Arena .venv inside this container fixes it.
#
# Usage:
#   ./run_arena_docker.sh '<bash command run inside the Arena venv>'
#   ./run_arena_docker.sh                 # drops into an interactive shell (venv active)
#
# Examples:
#   # L1 headless smoke (build LMDrillLiftRlD1, reset, step):
#   L1_HEADLESS=1 L1_STEPS=5 ./run_arena_docker.sh \
#     'python -u /home/juekunl/Work/arena_local_bringup/l1_viz_drilllift.py'
#
#   # Interactive poke-around:
#   ./run_arena_docker.sh
#
# IMPORTANT: always run Python with `python -u` (or PYTHONUNBUFFERED=1). Python block-
# buffers stdout when not a TTY, which hides print() output on Isaac Sim exit and makes
# a clean run look like a silent crash. This script sets PYTHONUNBUFFERED=1 for you.

set -euo pipefail

IMAGE="${ARENA_DOCKER_IMAGE:-nvcr.io/nvidian/gear-n2-eval:py312-2026.05.07}"
ARENA_PATH="${ISAACLAB_ARENA_PATH:-/home/juekunl/Work/IsaacLab-Arena}"
BRINGUP_PATH="${ARENA_BRINGUP_PATH:-/home/juekunl/Work/arena_local_bringup}"
GPU_DEV="${ARENA_GPU:-0}"
CACHE_ROOT="${ISAAC_CACHE_ROOT:-$HOME/docker/isaac-sim}"

# Persistent shader/asset caches: first run compiles shaders (slow, ~2.5 min),
# subsequent runs reuse them (~1 min). Mirrors gr00t docker/run_docker.sh.
mkdir -p "$CACHE_ROOT"/cache/{kit,ov,glcache,computecache} "$CACHE_ROOT"/{logs,data}

# Headless by default. Set L1_HEADLESS=0 (or GUI=1) to open the Isaac Sim window.
HEADLESS="${L1_HEADLESS:-1}"
if [[ "${GUI:-0}" == "1" ]]; then HEADLESS=0; fi

# Forward L1_* knobs (and any caller-exported ones) into the container.
ENV_FORWARD=(
  -e ACCEPT_EULA=Y
  -e OMNI_KIT_ALLOW_ROOT=1
  -e OMNI_KIT_ACCEPT_EULA=YES
  -e PYTHONUNBUFFERED=1
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e CUDA_VISIBLE_DEVICES=0
  -e ISAACLAB_ARENA_PATH="$ARENA_PATH"
  -e "L1_HEADLESS=${HEADLESS}"
  -e "L1_TASK=${L1_TASK:-LMDrillLiftRlD1}"
  -e "L1_STEPS=${L1_STEPS:-5}"
  -e "L1_ACTION=${L1_ACTION:-idle}"
  -e "L1_EMBODIMENT=${L1_EMBODIMENT:-}"
  -e "L1_VIDEO=${L1_VIDEO:-1}"
)

# GUI forwarding: when not headless, share the host X server so the Isaac Sim
# window appears on the host display. The container runs as root, so we must
# grant X access to local root (revoked again on exit).
X11_MOUNTS=()
if [[ "$HEADLESS" != "1" ]]; then
  HOST_DISPLAY="${DISPLAY:-:1}"
  if command -v xhost >/dev/null 2>&1; then
    xhost +local:root >/dev/null 2>&1 || true
    trap 'xhost -local:root >/dev/null 2>&1 || true' EXIT
  fi
  ENV_FORWARD+=(-e "DISPLAY=${HOST_DISPLAY}" -e "QT_X11_NO_MITSHM=1")
  X11_MOUNTS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  # Mount the active Xauthority if present (covers cookie-protected servers).
  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    X11_MOUNTS+=(-v "${XAUTHORITY}:${XAUTHORITY}:ro" -e "XAUTHORITY=${XAUTHORITY}")
  fi
  echo "[run_arena_docker] GUI mode: DISPLAY=${HOST_DISPLAY} (window will open on the host)"
fi

INNER_CMD="${1:-bash}"

# Only allocate a TTY when we actually have one (interactive terminal). This keeps
# the script usable from non-interactive contexts (CI, agents) without -it errors.
TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then TTY_FLAGS=(-it); fi

docker run --rm "${TTY_FLAGS[@]}" --gpus "\"device=${GPU_DEV}\"" --entrypoint /bin/bash \
  "${ENV_FORWARD[@]}" \
  "${X11_MOUNTS[@]}" \
  -v "$ARENA_PATH:$ARENA_PATH:rw" \
  -v "$BRINGUP_PATH:$BRINGUP_PATH:rw" \
  -v "$CACHE_ROOT/cache/kit:/isaac-sim/kit/cache:rw" \
  -v "$CACHE_ROOT/cache/ov:/root/.cache/ov:rw" \
  -v "$CACHE_ROOT/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "$CACHE_ROOT/cache/computecache:/root/.nv/ComputeCache:rw" \
  "$IMAGE" -lc "cd '$ARENA_PATH' && source .venv/bin/activate && ${INNER_CMD}"
