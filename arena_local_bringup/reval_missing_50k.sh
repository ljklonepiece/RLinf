#!/usr/bin/env bash
# Re-eval the 50k-sweep ckpts that crashed during concurrent IsaacSim init (NO_RESULT).
# Fix: 120s stagger (> ~40s sim-init) so only ONE sim initializes at a time (rollouts still
# run concurrently on the 8-GPU pool). Same eval: 100 eps, seed 42, video ON.
set -uo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKDIR=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k/drill_lift_bc_n1d7_tunevis_50k
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
OUT=/mnt/amlfs-07/shared/juekunl/eval_sweep_tunevis_50k
EVAL=/home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py
NGPU=8
MISSING="2000 4000 6000 8000 10000 12000 14000 18000 20000 24000"
declare -A GPU_PID

free_gpu () { for g in $(seq 0 $((NGPU-1))); do local p=${GPU_PID[$g]:-}; if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then echo "$g"; return; fi; done; }

echo "[reval] start $(date -u +%FT%TZ)  missing=$MISSING"
for S in $MISSING; do
  CK="$CKDIR/checkpoint-$S"
  G=""; while [ -z "$G" ]; do G=$(free_gpu); [ -z "$G" ] && sleep 20; done
  echo "[reval] checkpoint-$S on GPU $G ($(date -u +%T))"
  CUDA_VISIBLE_DEVICES="$G" timeout 7200 python -u "$EVAL" \
    --model_path "$CK" --backbone "$BB" --n_episodes 100 --seed 42 \
    --video_dir "$OUT/videos/checkpoint-$S" > "$OUT/eval_$S.log" 2>&1 &
  GPU_PID[$G]=$!
  sleep 120
done
wait
echo "===== REVAL RESULTS =====" | tee "$OUT/REVAL_SUMMARY.txt"
for S in $MISSING; do
  R=$(grep -h SUCCESS_RATE "$OUT/eval_$S.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/100 = [0-9.]+")
  echo "step $S: ${R:-STILL_FAILED}" | tee -a "$OUT/REVAL_SUMMARY.txt"
done
echo "REVAL_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/REVAL_SUMMARY.txt"
