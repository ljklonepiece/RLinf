#!/usr/bin/env bash
# Reproduction of the best 50k tune-visual recipe (identical to train_50k_tunevis.sh) but writing
# to *_repro output paths so the original run's checkpoints stay intact for comparison.
set -uxo pipefail
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export DATASET=/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl
export OUTPUT=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis_50k_repro
export EXPERIMENT_NAME=drill_lift_bc_n1d7_tunevis_50k_repro
export COLOR_JITTER_PARAMS="brightness 0.0 contrast 0.0 saturation 0.0 hue 0.0"
export STATE_DROPOUT_PROB=0.2
export MAX_STEPS=50000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=30
export LAUNCH_EXTRA="--tune-visual"
bash /workspace/bc_finetune_drill_lift_osmo.sh
