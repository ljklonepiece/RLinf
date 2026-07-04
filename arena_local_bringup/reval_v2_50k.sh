#!/usr/bin/env bash
# Robust re-eval of the 10 missing 50k-sweep ckpts. Prior attempts failed two ways:
#  - 20s stagger  -> concurrent IsaacSim INIT crashes (init > stagger)
#  - 120s stagger, 8-10 concurrent -> synchronized sim RESETS collided -> mid-rollout STALLS
# Fix: cap concurrency at MAXCC=4 (well below the ~7-8 that destabilized), one eval per distinct
# GPU, launches desynced by 60s. 100 eps, seed 42, video ON. Overwrites the bad eval_$S.log.
set -uo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKDIR=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k/drill_lift_bc_n1d7_tunevis_50k
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
O=/mnt/amlfs-07/shared/juekunl/eval_sweep_tunevis_50k
EVAL=/home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py
# Critical mid-range gaps first (their prior numbers were suspect), then the early undertrained ones.
STEPS="18000 20000 24000 14000 12000 10000 8000 6000 4000 2000"
MAXCC=4
declare -A GPU_PID

free_gpu () { for g in $(seq 0 7); do local p=${GPU_PID[$g]:-}; if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then echo "$g"; return; fi; done; }
n_running () { local n=0; for g in $(seq 0 7); do local p=${GPU_PID[$g]:-}; [ -n "$p" ] && kill -0 "$p" 2>/dev/null && n=$((n+1)); done; echo "$n"; }

echo "[reval2] start $(date -u +%FT%TZ) MAXCC=$MAXCC"
for S in $STEPS; do
  while [ "$(n_running)" -ge "$MAXCC" ]; do sleep 15; done
  G=$(free_gpu)
  echo "[reval2] checkpoint-$S on GPU $G ($(date -u +%T))"
  CUDA_VISIBLE_DEVICES="$G" timeout 14400 python -u "$EVAL" \
    --model_path "$CKDIR/checkpoint-$S" --backbone "$BB" --n_episodes 100 --seed 42 \
    --video_dir "$O/videos/checkpoint-$S" > "$O/eval_$S.log" 2>&1 &
  GPU_PID[$G]=$!
  sleep 60
done
wait
echo "===== REVAL2 RESULTS =====" | tee "$O/REVAL2_SUMMARY.txt"
for S in $STEPS; do
  R=$(grep -h SUCCESS_RATE "$O/eval_$S.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/100 = [0-9.]+")
  echo "step $S: ${R:-STILL_FAILED}" | tee -a "$O/REVAL2_SUMMARY.txt"
done
echo "REVAL2_DONE $(date -u +%FT%TZ)" | tee -a "$O/REVAL2_SUMMARY.txt"
