#!/usr/bin/env bash
# Autonomous eval sweep for the 50k tune-visual run. For each checkpoint (2000..50000 step 2000):
# wait until it is fully written, then eval it (100 eps, seed 42, video ON) on a rolling pool of
# 8 GPUs. SR logs + per-episode MP4s land on Lustre. Robust: skips ckpts already evaled
# (SUCCESS_RATE present), caps each eval at 2h (hang guard), staggers launches to avoid the
# concurrent sim-init hang. Meant to run detached (setsid nohup) on the eval cluster.
set -uo pipefail
cd /home/juekunl/Work/IsaacLab-Arena
source .venv/bin/activate
export MUJOCO_GL=egl
export PYTHONPATH=/home/juekunl/Work/RLinf:${PYTHONPATH:-}

CKDIR=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k/drill_lift_bc_n1d7_tunevis_50k
BB=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
OUT=/mnt/amlfs-07/shared/juekunl/eval_sweep_tunevis_50k
EVAL=/home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py
mkdir -p "$OUT/videos"
NGPU=8
EPS=100
declare -A GPU_PID

ckpt_ready () {  # $1=ckpt dir : index+2nd-shard exist and nothing modified in last 2 min (stable)
  [ -f "$1/model.safetensors.index.json" ] || return 1
  [ -f "$1/model-00002-of-00002.safetensors" ] || return 1
  [ -z "$(find "$1" -type f -mmin -2 2>/dev/null | head -1)" ]
}

free_gpu () {  # echo index of a GPU whose last eval finished (or was never used)
  for g in $(seq 0 $((NGPU-1))); do
    local p=${GPU_PID[$g]:-}
    if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then echo "$g"; return; fi
  done
}

echo "[orch] start $(date -u +%FT%TZ)  ckdir=$CKDIR"
for STEP in $(seq 2000 2000 50000); do
  CK="$CKDIR/checkpoint-$STEP"
  if grep -q SUCCESS_RATE "$OUT/eval_$STEP.log" 2>/dev/null; then echo "[skip] $STEP already evaled"; continue; fi
  echo "[wait] checkpoint-$STEP ready..."
  until ckpt_ready "$CK"; do sleep 30; done
  G=""; while [ -z "$G" ]; do G=$(free_gpu); [ -z "$G" ] && sleep 20; done
  echo "[launch] checkpoint-$STEP on GPU $G ($(date -u +%T))"
  CUDA_VISIBLE_DEVICES="$G" timeout 7200 python -u "$EVAL" \
    --model_path "$CK" --backbone "$BB" --n_episodes "$EPS" --seed 42 \
    --video_dir "$OUT/videos/checkpoint-$STEP" > "$OUT/eval_$STEP.log" 2>&1 &
  GPU_PID[$G]=$!
  sleep 20
done

wait
echo "===== SR-vs-STEPS (100 eps, seed 42) =====" | tee "$OUT/SUMMARY.txt"
for STEP in $(seq 2000 2000 50000); do
  R=$(grep -h SUCCESS_RATE "$OUT/eval_$STEP.log" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+ = [0-9.]+")
  echo "step $STEP: ${R:-NO_RESULT}" | tee -a "$OUT/SUMMARY.txt"
done
echo "ALL_EVAL_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/SUMMARY.txt"
