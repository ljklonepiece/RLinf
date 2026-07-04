#!/usr/bin/env bash
# Render eval videos for the BEST checkpoint (tune-visual @20k, ~59% SR) on the eval node.
# 6 GPUs, seeds 42-47, 8 episodes each => 48 rollouts. Then tar + upload to S3.
set -uxo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKPT=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis/drill_lift_bc_n1d7_tunevis/checkpoint-20000
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
OUT=/mnt/amlfs-07/shared/juekunl/vid_tunevis20k
rm -rf "$OUT"; mkdir -p "$OUT"
echo "OUT=$OUT"

EPS=8
i=0
for S in 42 43 44 45 46 47; do
  CUDA_VISIBLE_DEVICES=$i MUJOCO_GL=egl python -u /home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py \
    --model_path "$CKPT" --backbone "$BB" \
    --n_episodes "$EPS" --seed "$S" \
    --video_dir "$OUT/seed_$S" > "$OUT/log_seed_$S.log" 2>&1 &
  i=$((i+1))
done
wait

echo "===== PER-SEED RESULTS =====" | tee "$OUT/SUMMARY.txt"
for S in 42 43 44 45 46 47; do
  R=$(grep -h "SUCCESS_RATE" "$OUT/log_seed_$S.log" | tail -1)
  echo "seed $S: ${R:-NO_RESULT (see log_seed_$S.log)}" | tee -a "$OUT/SUMMARY.txt"
done
NVID=$(find "$OUT" -name "*.mp4" | wc -l)
echo "total mp4: $NVID" | tee -a "$OUT/SUMMARY.txt"

TAR=/tmp/vid_tunevis20k.tar
tar -C "$(dirname "$OUT")" -cf "$TAR" "$(basename "$OUT")"
ls -la "$TAR"
gear data upload "$TAR" s3://GearCheckpoints/juekunl/vid_tunevis20k/videos.tar --force 2>&1 | tail -5
echo "ALL_DONE_OK s3://GearCheckpoints/juekunl/vid_tunevis20k/videos.tar" | tee "$OUT/DONE.txt"
