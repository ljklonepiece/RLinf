#!/usr/bin/env bash
# Launch IsaacLab-Arena LOCALLY on the host venv with the Isaac Sim GUI window
# visible (display :1). Use this to WATCH the scene.
#
# Why host (not Docker) for GUI:
#   - The host venv GUI path renders a real window on the physical display :1.
#   - Docker X-forwarding connects to the X server but Kit's Vulkan WSI does not
#     reliably map a visible window from inside the container.
#   - The ONLY thing that crashed on the host was HEADLESS offscreen-camera render;
#     for headless runs use run_arena_docker.sh instead. GUI host runs are fine.
#
# Usage:
#   ./run_arena_host_gui.sh                 # 4000 steps, idle action (robot holds pose)
#   L1_ACTION=random L1_STEPS=2000 ./run_arena_host_gui.sh   # see it move
#
# Knobs (env vars): L1_TASK (default LMDrillLiftRlD1), L1_STEPS (4000),
#                   L1_ACTION (idle|zero|random), DISPLAY (default :1).

set -euo pipefail

ARENA_PATH="${ISAACLAB_ARENA_PATH:-/home/juekunl/Work/IsaacLab-Arena}"
BRINGUP_PATH="${ARENA_BRINGUP_PATH:-/home/juekunl/Work/arena_local_bringup}"

cd "$ARENA_PATH"
# shellcheck disable=SC1091
source .venv/bin/activate

export DISPLAY="${DISPLAY:-:1}"
export ISAACLAB_ARENA_PATH="$ARENA_PATH"
export ISAACLAB_PATH="$ARENA_PATH/submodules/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export L1_HEADLESS=0
export L1_TASK="${L1_TASK:-LMDrillLiftRlD1}"
export L1_STEPS="${L1_STEPS:-4000}"
export L1_ACTION="${L1_ACTION:-idle}"

echo "[run_arena_host_gui] DISPLAY=$DISPLAY task=$L1_TASK steps=$L1_STEPS action=$L1_ACTION"
echo "[run_arena_host_gui] a window titled 'Isaac Sim'/'Kit' should open on display $DISPLAY"
exec python -u "$BRINGUP_PATH/l1_viz_drilllift.py"
