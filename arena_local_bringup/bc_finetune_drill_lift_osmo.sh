#!/usr/bin/env bash
# BC finetune of PUBLIC GR00T-N1.7-3B on IsaacLab-Arena G1 drill_lift, via the OFFICIAL
# Isaac-GR00T wrapper examples/finetune.sh (launch_finetune.py + torchrun).
#
# RUN LOCATION: an OSMO H100 node, inside the public Isaac-GR00T checkout (/workspace/Isaac-GR00T)
# whose uv env is created with `uv sync`. Invoked via `uv run`.
#
# ============================ PER-CLUSTER NODE SETUP (idempotent) =========================
# /workspace is EPHEMERAL (new on every cluster); Lustre /mnt/amlfs-07/... PERSISTS.
#  1. Clone + x86_64 uv env (the only slow step, ~10 min):
#       GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules --depth 1 \
#         https://github.com/NVIDIA/Isaac-GR00T /workspace/Isaac-GR00T
#     then in pyproject.toml: `[tool.uv] required-environments` -> `environments` (x86_64 only),
#     delete the two aarch64 path-wheel sources under scripts/deployment/dgpu/wheels/ and the
#     now-empty `torchcodec = []` block; then `uv sync`.  (uv otherwise tries to read the
#     LFS-pointer aarch64 wheels and fails; we only run x86_64.)
#  2. Source patches (this script runs them): osmo_patch_isaac_gr00t.py — env overrides for
#     backbone path / action_horizon / skip_first_n_frames, the skip-first-frame data fix, and
#     SAVE_TOTAL_LIMIT env in finetune.sh.
#  3. Lustre assets (persist; this script re-asserts the config.json edits idempotently):
#       - base model GR00T-N1.7-3B config.json: model_name -> local nvidia/Cosmos-Reason2 path,
#         action_horizon -> 50 (G1 canonical; NO weight resize — pos-embed is sized to max_seq_len)
#       - staged gated backbone at .../models/nvidia/Cosmos-Reason2-2B (no HF token needed)
# =========================================================================================
#
# WHY skip_first_n_frames=1: the teleop/rollout recorder logs a STALE first frame per episode
# (off-by-one between observation and action/reward), so the first transition is corrupted.
# Matches the internal gr00t offline-RL pipeline's `--skip-first-n-frames 1`. Verified: drops
# exactly one step/episode (drill_lift_rl: 33802 -> 33687 total steps over 115 episodes).
#
# WHY UNITREE_G1: the public base ships EmbodimentTag.UNITREE_G1
# ("unitree_g1_full_body_with_waist_height_nav_cmd"); its modality == our drill_lift_rl dataset.
#
# SETTINGS from examples/GR00TWholeBodyControl (G1 WBC single-task ~150 eps: 20k iters @ batch 32).
# Checkpoints: --save-only-model (model weights only, ~6.5GB) + SAVE_TOTAL_LIMIT=10 + save every
# 2k -> keep the FULL 2k..20k sweep for S4.4 selection (the best BC ckpt is often NOT the last).
#
# USAGE (on the node):
#   DRY_RUN=1 FAST=1 bash bc_finetune_drill_lift_osmo.sh   # ~2-min pipeline check (skip ckpt load)
#   DRY_RUN=1 bash bc_finetune_drill_lift_osmo.sh          # 1-step, real weights
#   bash bc_finetune_drill_lift_osmo.sh                    # full 20k-step finetune (8 GPUs)
set -euo pipefail

REPO="${ISAAC_GR00T_REPO:-/workspace/Isaac-GR00T}"
PATCH_SCRIPT="${PATCH_SCRIPT:-/workspace/osmo_patch_isaac_gr00t.py}"
MODEL=/mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B
BACKBONE=/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
# WITH-EEF experiment: drill_lift_rl_wbc_eef_v2 has observation.eef_pose (pelvis-relative
# wrist pos+quat) as the state keys left_wrist_pose/right_wrist_pose (added to the UNITREE_G1
# state modality by osmo_patch_isaac_gr00t.py). Overridable via env for the no-eef baseline.
DATASET="${DATASET:-/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl_wbc_eef_v2}"
OUTPUT="${OUTPUT:-/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_eef}"

# (2) source patches (idempotent)
[[ -f "${PATCH_SCRIPT}" ]] && python3 "${PATCH_SCRIPT}" "${REPO}" || echo "[warn] ${PATCH_SCRIPT} missing; assuming repo already patched"

# (3) Lustre base config.json: local backbone + horizon 50 (idempotent; keeps config.json.bak40)
python3 - "$MODEL" "$BACKBONE" <<'PY'
import json, sys, pathlib
cfg = pathlib.Path(sys.argv[1]) / "config.json"
bak = cfg.parent / "config.json.bak40"
d = json.loads(cfg.read_text())
if not bak.exists(): bak.write_text(json.dumps(d, indent=2))
d["model_name"] = sys.argv[2]; d["action_horizon"] = 50
cfg.write_text(json.dumps(d, indent=2))
print(f"[setup] base config: model_name={d['model_name']} action_horizon={d['action_horizon']}")
PY

export HF_HOME="${HF_HOME:-/mnt/amlfs-07/shared/groot_oss/ci/hf_home}"
export GR00T_BACKBONE_PATH="${GR00T_BACKBONE_PATH:-$BACKBONE}"
export GR00T_ACTION_HORIZON="${GR00T_ACTION_HORIZON:-50}"
export GR00T_SKIP_FIRST_N_FRAMES="${GR00T_SKIP_FIRST_N_FRAMES:-1}"   # drop stale first frame/episode

export NUM_GPUS="${NUM_GPUS:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export USE_WANDB="${USE_WANDB:-0}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

EXTRA=(--save-only-model)   # keep the full ckpt sweep cheaply (weights only; we eval+freeze, not resume)
# finetune.sh hardcodes STATE_DROPOUT_PROB="" (ignores the env), so pass it as a real CLI arg.
[[ -n "${STATE_DROPOUT_PROB:-}" ]] && EXTRA+=(--state-dropout-prob "${STATE_DROPOUT_PROB}")
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  export NUM_GPUS=1 MAX_STEPS=1 SAVE_STEPS=1 GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE_DRY:-2}"
  OUTPUT="${OUTPUT}_dryrun"
  [[ "${FAST:-0}" == "1" ]] && EXTRA+=(-- --skip_weight_loading)
  echo "[bc-finetune] DRY RUN (FAST=${FAST:-0}) skip=${GR00T_SKIP_FIRST_N_FRAMES} -> $OUTPUT"
else
  export MAX_STEPS="${MAX_STEPS:-20000}" SAVE_STEPS="${SAVE_STEPS:-2000}"
  echo "[bc-finetune] FULL RUN: ${MAX_STEPS} steps, ${NUM_GPUS} GPUs, batch ${GLOBAL_BATCH_SIZE}, skip=${GR00T_SKIP_FIRST_N_FRAMES}, keep ${SAVE_TOTAL_LIMIT} ckpts -> $OUTPUT"
fi

# extra launch_finetune.py passthrough args (after --), e.g. LAUNCH_EXTRA="--tune-visual"
[[ -n "${LAUNCH_EXTRA:-}" ]] && EXTRA+=(-- ${LAUNCH_EXTRA})

cd "${REPO}"
exec uv run bash examples/finetune.sh \
  --base-model-path "${MODEL}" \
  --dataset-path "${DATASET}" \
  --embodiment-tag UNITREE_G1 \
  --output-dir "${OUTPUT}" \
  --experiment-name "${EXPERIMENT_NAME:-drill_lift_bc_n1d7_eef}" \
  "${EXTRA[@]}"
