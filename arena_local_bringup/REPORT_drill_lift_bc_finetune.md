# Finetuning a BC policy on the public Isaac-GR00T N1.7 base — drill_lift findings

**Author:** juekunl · **Date:** 2026-07-02 · **Task:** IsaacLab-Arena G1 `drill_lift` (lift the power drill), pure behavior cloning (no RL) on the **public** `nvidia/GR00T-N1.7-3B`.

---

## TL;DR

- A well-trained public GR00T-N1.7 BC policy reaches **59% success on drill_lift** — essentially matching the internal **iter0-IQL-on-Eagle baseline (60–70%)**, using **pure BC** on the public Cosmos-Reason2 backbone (no RL, no internal model).
- The earlier **0%** result was **not a fundamental gap** — it was a **recipe/undertraining artifact**. Three levers, in order of impact:
  1. **Train the full 20k steps** (the 0% run was reclaimed at ~10k → undertrained). *Dominant.*
  2. **Unfreeze the visual encoder** (`--tune-visual`): **30% → 59%** (~2×). *Biggest recipe lever; not used in any official Isaac-GR00T recipe.*
  3. **Turn color-jitter off**: **14% → 30%** (~2×). *Sim task — the sim2real jitter default hurts.*
- **Best checkpoint (59%):**
  `/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis/drill_lift_bc_n1d7_tunevis/checkpoint-20000`

---

## Results

Eval: `LMDrillLiftRlD1` (Arena G1), **100 episodes, seed 42**, success = lift 0.15 m + grasped, `n_action_steps=8`, `max_episode_steps=720`, terminate-on-success — identical methodology to the internal gr00t sim eval, so numbers are comparable.

| # | Recipe (all: public N1.7, BC, `drill_lift_rl`, batch 32, lr 1e-4) | @16k | @18k | @20k |
|---|---|---|---|---|
| 0 | baseline: frozen visual, jitter ON, **reclaimed at ~10k** | — | — | **0%** |
| 1 | frozen visual, **jitter OFF**, 20k | 9% | — | **30%** |
| 2 | frozen visual, jitter ON, 20k (fair control) | 16% | — | **14%** |
| 3 | **visual UNFROZEN**, jitter OFF, 20k | **47%** | **54%** | **59%** |
| — | *internal reference: iter0-IQL on Eagle backbone* | — | — | *60–70%* |

Both recipes are ~0% at ≤12k and only begin succeeding at 16k+, which is why the original ~10k run scored 0%.

---

## Why we trust this (diagnostics)

Before re-training, we verified the 0% was not a bug in our eval:
- **Checkpoint learned the task open-loop** — action-reconstruction MSE ≈ 0.005–0.008 (both Isaac-GR00T native path and our RLinf `gr00t_n1d7` loader agree).
- **Plumbing verified correct** — sim state (43-d joints), camera (`robot_head_cam_rgb`→`ego_view`, uint8), 35-d action layout, and env (`LMDrillLiftRlD1`) all match the training data / internal eval.
- **Base weights identical** — the public `GR00T-N1.7-3B` is **bit-identical** to the internal `n17_es_cr2_...` base (all 1031 tensors, max-diff 0). So the base model is not the differentiator.
- **Fair, step-matched control** (EXP2) isolated color-jitter (~2×) from training duration (dominant).

---

## How to reproduce

Everything runs on an OSMO H100 cluster; Lustre (`/mnt/amlfs-07/shared/juekunl/`) holds the base model, backbone, dataset, and checkpoints. Scripts are in `arena_local_bringup/`.

**1. Provision + deploy the env on a fresh node** (clone Isaac-GR00T @ `ab88b50`, x86 pyproject fix, `uv sync`, source patches):
```bash
# provision (2-node H100):
cd ~/Work/gr00t && uv run gear ray start-cluster -n 2 -p groot-h100-01 -pr HIGH -w drilllift_bc
# push scripts + deploy (~5 min):
cd ~/Work/arena_local_bringup
bash push_to_node.sh <workflow_id> master \
  deploy_finetune_node.sh:/workspace/deploy_finetune_node.sh \
  osmo_patch_isaac_gr00t.py:/workspace/osmo_patch_isaac_gr00t.py \
  bc_finetune_drill_lift_osmo.sh:/workspace/bc_finetune_drill_lift_osmo.sh
bash osmo_exec.sh <workflow_id> master 'bash /workspace/deploy_finetune_node.sh'   # -> DEPLOY_DONE_OK
```

**2. Train the BEST recipe (tune-visual, ~2.8 h on 8×H100):**
```bash
bash osmo_exec.sh <workflow_id> master '
cd /workspace
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
DATASET=/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl \
OUTPUT=/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis \
EXPERIMENT_NAME=drill_lift_bc_n1d7_tunevis \
COLOR_JITTER_PARAMS="brightness 0.0 contrast 0.0 saturation 0.0 hue 0.0" \
STATE_DROPOUT_PROB=0.2 SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=20 LAUNCH_EXTRA="--tune-visual" \
setsid nohup bash /workspace/bc_finetune_drill_lift_osmo.sh > /workspace/tunevis.log 2>&1 < /dev/null &'
```
Recipe = Isaac-GR00T `finetune.sh` (lr 1e-4, warmup 0.05, wd 1e-5, batch 32, freeze LLM, tune projector+diffusion) **plus**: `--tune-visual` (unfreeze visual), color-jitter 0, `action_horizon=50`, local Cosmos-Reason2-2B backbone, `skip_first_n_frames=1`, 20k steps.
- **Required:** `NCCL_P2P_DISABLE=1` — without it, tune-visual crashes with a NCCL/NVLink peer-access error on these nodes.

**3. Evaluate (100 eps, seed 42):**
```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python ~/Work/arena_local_bringup/eval_bc_ckpt.py \
  --model_path /mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis/drill_lift_bc_n1d7_tunevis/checkpoint-20000 \
  --backbone /mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B \
  --n_episodes 100 --seed 42
# or the parallel 8-GPU sweep: eval_sweep_osmo.yaml
```

To reproduce EXP1 (no-jitter, 30%): same as above **without** `LAUNCH_EXTRA` and with a different OUTPUT dir.

---

## Best checkpoint

| | path |
|---|---|
| **Best (59%)** | `/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7_tunevis/drill_lift_bc_n1d7_tunevis/checkpoint-20000` |
| no-jitter (30%) | `.../drill_lift_bc_gr00t_n1d7_nojitter/drill_lift_bc_n1d7_nojitter/checkpoint-20000` |
| base model | `/mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B` (Cosmos-Reason2-2B backbone staged at `.../models/nvidia/Cosmos-Reason2-2B`) |
| dataset | `/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl` (115 successful demos, 50 fps) |

Load the best ckpt in RLinf via the `gr00t_n1d7` loader with `embodiment_tag=UNITREE_G1` / `obs_converter_type=isaaclab_arena_g1`, `action_dim=35`, `action_horizon=50`.

---

## Caveats & next steps
- **59% is pure BC** and already ≈ the internal iter0-IQL (60–70%). RL/DSRL on top of this frozen base is the natural next step to push higher.
- **tune-visual still rising at 20k** (47→54→59) — more steps and/or larger batch (official robocasa recipes use 60k/batch-512) may yield further gains.
- **NCCL_P2P_DISABLE is required** for tune-visual on the current H100 pools (NVLink peer-access fault otherwise); it costs throughput (~2.3 it/s).
- Full experiment log: `RLinf/rlinf/envs/isaaclab_arena/EXPERIMENTS.md`.
