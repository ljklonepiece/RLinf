#!/usr/bin/env bash
# 50k-step tune-visual (no color jitter) BC finetune: the BEST-ckpt recipe, extended 20k -> 50k,
# saving every 2000 steps (25 ckpts) for a full SR-vs-steps sweep. Ground-truth recipe taken from
# checkpoint-20000/experiment_cfg: tune_visual=true, color_jitter all 0, state_dropout=0.2,
# lr 1e-4 cosine, wd 1e-5, warmup 0.05, batch 32, bf16, deepspeed-2, 8 GPU, action_horizon 50,
# skip_first_n_frames 1, save_only_model, dataset drill_lift_rl (no-eef).
#
# NOTE: finetune.sh DEFAULTS color-jitter to ON (0.3/0.4/0.5/0.08); we MUST export
# COLOR_JITTER_PARAMS=all-zero to reproduce the no-jitter best recipe. NCCL_P2P_DISABLE is
# required or tune-visual crashes with an NVLink peer-access fault on these nodes.
set -uxo pipefail
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export DATASET=/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl
export OUTPUT=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k
export EXPERIMENT_NAME=drill_lift_bc_n1d7_tunevis_50k
export COLOR_JITTER_PARAMS="brightness 0.0 contrast 0.0 saturation 0.0 hue 0.0"
export STATE_DROPOUT_PROB=0.2
export MAX_STEPS=50000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=30
export LAUNCH_EXTRA="--tune-visual"
bash /workspace/bc_finetune_drill_lift_osmo.sh
