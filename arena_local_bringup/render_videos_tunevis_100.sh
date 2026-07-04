#!/usr/bin/env bash
# Faithful 100-episode seed-42 render for the BEST checkpoint (tune-visual @20k).
# Single GPU (episodes are sequential under one seed). Then tar + upload to S3.
set -uxo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKPT=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis/drill_lift_bc_n1d7_tunevis/checkpoint-20000
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
OUT=/mnt/amlfs-07/shared/juekunl/vid_tunevis20k_seed42_100
rm -rf "$OUT"; mkdir -p "$OUT"
echo "OUT=$OUT"

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -u /home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py \
  --model_path "$CKPT" --backbone "$BB" \
  --n_episodes 100 --seed 42 \
  --video_dir "$OUT/seed_42" > "$OUT/log_seed_42.log" 2>&1
echo "eval rc=$?"

grep -h SUCCESS_RATE "$OUT/log_seed_42.log" | tail -1 | tee "$OUT/SUMMARY.txt"
find "$OUT" -name "*.mp4" | wc -l | sed 's/^/total mp4: /' | tee -a "$OUT/SUMMARY.txt"

TAR=/tmp/vid_tunevis20k_seed42_100.tar
tar -C "$(dirname "$OUT")" -cf "$TAR" "$(basename "$OUT")"
ls -la "$TAR"
gear data upload "$TAR" s3://GearCheckpoints/juekunl/vid_tunevis20k/videos_seed42_100.tar --force 2>&1 | tail -5
echo "ALL_DONE_OK s3://GearCheckpoints/juekunl/vid_tunevis20k/videos_seed42_100.tar" | tee "$OUT/DONE.txt"
