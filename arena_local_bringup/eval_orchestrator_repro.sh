#!/usr/bin/env bash
# Robust ckpt-waiting eval sweep for the REPRO 50k run. Waits for each checkpoint as training
# produces it, then evals 100 eps / seed 42 / video on a rolling pool capped at MAXCC=4 (avoids the
# concurrent sim-init crashes AND reset stalls seen in the original run), 120s stagger, 240-min
# per-eval timeout (no truncation). Uses the baked RLinf on the eval image.
set -uo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKDIR=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k_repro/drill_lift_bc_n1d7_tunevis_50k_repro
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
OUT=/mnt/amlfs-07/shared/juekunl/eval_sweep_tunevis_50k_repro
EVAL=/home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py
mkdir -p "$OUT/videos"
NGPU=8; MAXCC=4; EPS=100
declare -A GPU_PID

ckpt_ready(){ [ -f "$1/model.safetensors.index.json" ]||return 1; [ -f "$1/model-00002-of-00002.safetensors" ]||return 1; [ -z "$(find "$1" -type f -mmin -2 2>/dev/null|head -1)" ]; }
free_gpu(){ for g in $(seq 0 $((NGPU-1))); do local p=${GPU_PID[$g]:-}; if [ -z "$p" ]||! kill -0 "$p" 2>/dev/null; then echo "$g"; return; fi; done; }
n_running(){ local n=0; for g in $(seq 0 $((NGPU-1))); do local p=${GPU_PID[$g]:-}; [ -n "$p" ]&&kill -0 "$p" 2>/dev/null&&n=$((n+1)); done; echo $n; }

echo "[repro-orch] start $(date -u +%FT%TZ)  ckdir=$CKDIR"
for STEP in $(seq 2000 2000 50000); do
  CK="$CKDIR/checkpoint-$STEP"
  grep -q SUCCESS_RATE "$OUT/eval_$STEP.log" 2>/dev/null && { echo "[skip] $STEP done"; continue; }
  echo "[wait] checkpoint-$STEP ..."; until ckpt_ready "$CK"; do sleep 30; done
  while [ "$(n_running)" -ge "$MAXCC" ]; do sleep 15; done
  G=$(free_gpu)
  echo "[launch] checkpoint-$STEP on GPU $G ($(date -u +%T))"
  CUDA_VISIBLE_DEVICES="$G" timeout 14400 python -u "$EVAL" \
    --model_path "$CK" --backbone "$BB" --n_episodes "$EPS" --seed 42 \
    --video_dir "$OUT/videos/checkpoint-$STEP" > "$OUT/eval_$STEP.log" 2>&1 &
  GPU_PID[$G]=$!
  sleep 120
done
wait
echo "===== REPRO SR-vs-STEPS (100 eps, seed 42) =====" | tee "$OUT/SUMMARY.txt"
for STEP in $(seq 2000 2000 50000); do
  R=$(grep -h SUCCESS_RATE "$OUT/eval_$STEP.log" 2>/dev/null|tail -1|grep -oE "[0-9]+/100 = [0-9.]+")
  echo "step $STEP: ${R:-NA}" | tee -a "$OUT/SUMMARY.txt"
done
echo "REPRO_EVAL_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/SUMMARY.txt"
