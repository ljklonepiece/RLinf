# Runbook: BC train + eval of GR00T-N1.7 on IsaacLab-Arena drill_lift (OSMO)

How to reproduce the full **train → checkpoint sweep → eval (with video)** pipeline on *any* fresh
OSMO workflow. Workflow IDs change every time; everything below is parameterized by `<WF>` (the
workflow id) and `<TASK>` (always `master` for the gear ray clusters we use).

All persistent assets live on Lustre `/mnt/amlfs-07/shared/juekunl/` (mounted on every node);
`/workspace` is **ephemeral** (re-created per cluster, so the env must be re-deployed each time).

---

## 1. Artifacts (in `~/Work/RLinf/arena_local_bringup/`, git-tracked on branch `jk/isaaclab-arena-gr00t`)

| script | runs where | purpose |
|---|---|---|
| `osmo_exec.sh <WF> <TASK> '<cmd>'` | local | run a shell command on a workflow node (PTY via `script`, base64-wrapped) |
| `push_to_node.sh` | local | chunked file push (⚠ flaky — prefer the atomic base64 push in §4) |
| `deploy_finetune_node.sh` | node | clone+build Isaac-GR00T env on a fresh train node (idempotent) |
| `osmo_patch_isaac_gr00t.py` | node | source patches: backbone path, action_horizon 50, skip_first_n_frames, SAVE_TOTAL_LIMIT env |
| `bc_finetune_drill_lift_osmo.sh` | node | BC finetune wrapper around official `examples/finetune.sh` |
| `train_50k_tunevis.sh` | node | best-recipe launcher (tune-visual, no-jitter, 50k, save every 2000) |
| `eval_bc_ckpt.py` | node | eval one ckpt (RLinf `gr00t_n1d7` loader, 100 eps, seed 42, video default-ON) |
| `eval_orchestrator_50k.sh` | node | auto-eval each ckpt as it appears (rolling GPU pool) — use *during* training |
| `reval_v2_50k.sh` | node | re-eval a fixed ckpt list at controlled concurrency — use *after* training |

---

## 2. Canonical paths

```
# Lustre (persistent, on every node)
BASE_MODEL   = /mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B        # config.json patched: model_name->local Cosmos, action_horizon 50
BACKBONE     = /mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B
DATASET      = /mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl   # no-eef, 115 demos (the eef v2 set is drill_lift_rl_wbc_eef_v2)
CKPTS        = /mnt/amlfs-07/shared/juekunl/ckpts/<exp>/<exp_name>/checkpoint-<N>   # ~6.5GB each (save-only-model)
EVAL_OUT     = /mnt/amlfs-07/shared/juekunl/eval_sweep_<name>/          # eval_<N>.log, videos/checkpoint-<N>/, SUMMARY.txt

# Node env (baked by deploy)
ISAAC_GR00T  = /workspace/Isaac-GR00T                 # uv env; invoke via `uv run`
ARENA_VENV   = /home/juekunl/Work/IsaacLab-Arena/.venv  # eval env (arena 4.6.5 + RLinf on PYTHONPATH)
RLINF        = /home/juekunl/Work/RLinf               # add to PYTHONPATH for eval

# S3 relay (endpoint https://pdx.s8k.io)
S3_BUCKET    = s3://GearCheckpoints/juekunl/          # our scratch bucket for node<->local transfer
```

---

## 3. Provision clusters (gear)

```bash
cd ~/Work/gr00t && source .venv/bin/activate
# Train: 8xH100 (2 nodes). Eval: 8xL40 (1 node). -w sets a name fragment; the real id is printed.
uv run gear ray start-cluster -n 1 -p groot-h100-01 -pr HIGH -w drilllift_train   # 1 node = 8xH100 (see note)
uv run gear ray start-cluster -n 1 -p <l40-pool>     -pr HIGH -w drilllift_eval    # -> eval_ray_cluster_1n_uuid_XXXX-1
uv run gear ray list-clusters   # or: osmo workflow list   (find your <WF> id + status RUNNING)
```
Reuse an existing idle RUNNING cluster instead of provisioning when possible (deploy persists on it).

> **Note (train = single node):** the finetune is single-node `torchrun --nproc_per_node=8` (NUM_GPUS=8), so
> **`-n 1` is correct** — one H100 node = 8 GPUs is fully used. Using `-n 2` provisions a second node whose 8
> GPUs sit **idle** (this script has no multi-node path); don't do it.
>
> **Note (reclamation resilience):** HIGH-priority H100 pools get reclaimed roughly once per multi-hour run
> (observed on the original nojitter run AND the repro, which died at step 30k/50k). Because finetune uses
> `--save-only-model` (no optimizer/scheduler state), there is **no clean training resume** — a reclaim means
> re-running from scratch. Mitigations: run at the highest priority available, keep `SAVE_STEPS` frequent so
> you at least keep the checkpoints produced so far, and be ready to re-provision + relaunch.

---

## 4. OSMO node primitives (the important mechanics)

```bash
# Run a command on a node:
./osmo_exec.sh <WF> master 'hostname; ls /mnt/amlfs-07/shared/juekunl'

# Atomic file push (RELIABLE for files whose base64 < ~9KB; verify md5). Prefer this over push_to_node.sh:
LMD5=$(md5sum myscript.sh|awk '{print $1}'); B64=$(base64 -w0 myscript.sh)
./osmo_exec.sh <WF> master "printf '%s' '$B64' | base64 -d > /workspace/myscript.sh; \
  md5sum /workspace/myscript.sh   # compare to $LMD5"

# Run a long job DETACHED (plain nohup dies when the exec session ends — use setsid):
./osmo_exec.sh <WF> master 'cd /workspace; setsid nohup bash myscript.sh > /tmp/job.log 2>&1 < /dev/null & echo started'

# Transfer node <-> local via S3 (⚠ node `gear` is BROKEN: "No module named sky" -> use s5cmd on the node):
#   creds (fetch fresh; they rotate):
eval "$(gear data get-credentials)"    # sets AWS_* + AWS_ENDPOINT_URL locally
#   node -> S3 (on node, s5cmd is at /usr/local/bin/s5cmd):
./osmo_exec.sh <WF> master "export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...; \
  s5cmd --endpoint-url https://pdx.s8k.io cp /tmp/out.tar s3://GearCheckpoints/juekunl/out.tar"
#   S3 -> local:
/home/juekunl/bin/s5cmd --endpoint-url https://pdx.s8k.io cp s3://GearCheckpoints/juekunl/out.tar ./out.tar
#   (local `gear data download s3://... <dir>` also works, but treats dst as a DIR and needs a user-owned dir)

# Keepalive: idle clusters run a Ray `manual_keepalive` gpu_burner to avoid reclamation.
#   Kill before your own GPU work, restart when leaving the cluster idle:
./osmo_exec.sh <WF> master 'ray job list | grep -o "manual_keepalive_[a-z0-9]*"'   # find id
./osmo_exec.sh <WF> master 'ray job stop <submission_id>'
```

---

## 5. Deploy env on a fresh TRAIN node (idempotent, ~10-15 min)

```bash
cd ~/Work/arena_local_bringup
for f in deploy_finetune_node.sh osmo_patch_isaac_gr00t.py bc_finetune_drill_lift_osmo.sh train_50k_tunevis.sh; do
  LMD5=$(md5sum $f|awk '{print $1}'); B64=$(base64 -w0 $f)
  ./osmo_exec.sh <WF> master "printf '%s' '$B64'|base64 -d > /workspace/$f; \
    [ \"\$(md5sum /workspace/$f|awk '{print \$1}')\" = \"$LMD5\" ] && echo OK $f || echo BAD $f"
done
./osmo_exec.sh <WF> master 'bash /workspace/deploy_finetune_node.sh'   # -> DEPLOY_DONE_OK
```
(Isaac-GR00T pinned `ab88b50`; deploy does pyproject x86-only edits + `uv sync` + source patches. Lustre
base-model `config.json` edits are re-asserted by `bc_finetune_drill_lift_osmo.sh`.)

---

## 6. TRAIN (best recipe: tune-visual, no color-jitter)

Ground-truth recipe (from a best-ckpt `experiment_cfg/config.yaml`): `tune_visual=true`, color_jitter all 0,
`state_dropout=0.2`, lr 1e-4 cosine, wd 1e-5, warmup 0.05, batch 32, bf16, deepspeed-2, 8 GPU,
action_horizon 50, skip_first_n_frames 1, save-only-model.

```bash
./osmo_exec.sh <WF> master 'cd /workspace; setsid nohup bash train_50k_tunevis.sh > /tmp/train.log 2>&1 < /dev/null & echo started'
# train_50k_tunevis.sh exports (edit these for other runs):
#   DATASET / OUTPUT / EXPERIMENT_NAME / MAX_STEPS / SAVE_STEPS / SAVE_TOTAL_LIMIT
#   COLOR_JITTER_PARAMS="brightness 0.0 contrast 0.0 saturation 0.0 hue 0.0"   # ⚠ MUST set — finetune.sh defaults jitter ON (0.3/0.4/0.5/0.08)
#   STATE_DROPOUT_PROB=0.2 ; LAUNCH_EXTRA="--tune-visual"
#   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1   # ⚠ REQUIRED for --tune-visual (else NVLink P2P CUDA crash)
```
Rate ~2.3 it/s on 8xH100 → 50k steps ≈ 6h. Monitor: `tail /tmp/train.log`; ckpts land under `$OUTPUT/$EXPERIMENT_NAME/checkpoint-<N>`.
Free Lustre first: 25 ckpts × 6.5GB ≈ 162GB (`stat -f` to check; `lfs` unavailable).

---

## 7. EVAL (100 eps, seed 42, video ON) + collect

Eval runs in the ARENA venv, not the Isaac-GR00T uv env:
```bash
# single ckpt:
./osmo_exec.sh <WF> master 'cd /home/juekunl/Work/IsaacLab-Arena; source .venv/bin/activate; \
  MUJOCO_GL=egl PYTHONPATH=/home/juekunl/Work/RLinf CUDA_VISIBLE_DEVICES=0 \
  python -u /home/juekunl/Work/arena_local_bringup/eval_bc_ckpt.py \
    --model_path <CKPT> --backbone /mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B \
    --n_episodes 100 --seed 42'    # video auto-saved to ./eval_videos/<ckpt>/seed_42; --no_video to disable
```
- **Sweep during training:** push+launch `eval_orchestrator_50k.sh` (edit CKDIR/OUT) — waits for each ckpt then evals on a rolling 8-GPU pool.
- **Sweep after training:** push+launch `reval_v2_50k.sh` (edit STEPS/CKDIR/OUT) — evals a fixed list.
- **Collect videos:** on node `tar -C <OUT/..> -cf /tmp/v.tar <dir>` → `s5cmd cp` to S3 → pull locally (§4). Each episode = 1 MP4 (~1.5MB), overlaid with reward+termination.

Success criterion = Arena's own terminal for `LMDrillLiftRlD1` (lift 0.15 + grasped), identical to the internal
IQL eval, so numbers are comparable. Env source: `mmoghani/IsaacLab-Arena`, branch `darrelldai/offline-rl-new`.

---

## 8. Gotchas / lessons learned (READ before an eval sweep)

- **Concurrent sim INIT crashes:** if eval launches are staggered < ~40s (the sim-init time), simultaneous
  IsaacSim inits collide → process dies at init ("leaked semaphore at shutdown", no result). Stagger ≥120s
  *or* keep concurrency low.
- **Synchronized sim RESETS stall:** bursting many evals so they run in lockstep makes their per-episode resets
  collide → mid-rollout stalls/hangs. Cap concurrency (**MAXCC≈4** was stable; 8 caused stalls). Launches
  spaced ~14 min apart (as in the training-gated orchestrator) also avoid it.
- **Per-eval timeout must exceed 100eps × the *concurrent* rate.** Solo ≈0.9 min/ep (~90 min); MAXCC=4 ≈1.6
  min/ep (~160 min). Use `timeout 14400` (240 min). A too-short timeout silently kills evals at ~ep 85 → NO_RESULT.
- **Concurrency does NOT bias completed-episode SR.** Each eval is 1 env on its own GPU with fixed physics dt,
  so concurrency only causes hangs/slowdowns; episodes that finish are trustworthy (verify failures are real
  720-step timeouts, not short/broken episodes).
- **Node `gear` is broken** (missing `sky`) → use `s5cmd` on the node for S3.
- **`setsid nohup`** for any node job that must outlive the `osmo_exec` session.
- **`--tune-visual` needs `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`** on these H100 nodes.
- **Verify a completed eval reached 100 eps**: `grep -c 'ep .*/100:' eval_<N>.log` == 100 and a `X/100` SUCCESS_RATE line.
- **tune-visual is unstable**: task SR swings wildly across adjacent checkpoints (0%↔74%) — always sweep many
  ckpts and pick the best; the last checkpoint is often not the best.

---

## 9. Repository provenance (git state used for these results)

Captured 2026-07-03. **⚠ Several repos carry uncommitted changes that are essential to reproduce this
pipeline — the commit hash alone is NOT enough.** See caveats below.

| repo | local path | origin | branch | commit | working tree |
|---|---|---|---|---|---|
| IsaacLab-Arena | `~/Work/IsaacLab-Arena` | `gitlab-master:mmoghani/IsaacLab-Arena.git` | `darrelldai/offline-rl-new` (**detached**) | `dd6618c682ecc24536a2a75f80a2ac3e6b879d64` | clean |
| RLinf | `~/Work/RLinf` | `github.com/RLinf/RLinf.git` | `main` | `f711061e4195da98c89983a236b2328a09d914fd` | **15 uncommitted** (Arena integration) |
| Isaac-GR00T (train) | `~/Work/Isaac-GR00T` (node re-clones) | `github.com/NVIDIA/Isaac-GR00T.git` | `main` | `ab88b50c718f6528e1df9dcbaf75865d1b604760` | 5 modified = the `osmo_patch` edits |
| gr00t (gear + baseline) | `~/Work/gr00t` | `gitlab-master:GR00T/gr00t/gr00t.git` | `jk/offline-rl` | `cf234bd4075c66ecba0aa038fc6b3058baf9cf14` | 7 untracked (offline-rl helpers) |

**IsaacLab-Arena submodules** (pinned at the commit above):
- `assets` → `b17fdc328f09bc8b1df8bfa07cef06ad37dbd013`
- `submodules/Isaac-GR00T` → `3bce5530b3af0ce3619d5fda041385f9b732dca8` (`n1.5-release~18`)
- `submodules/IsaacLab` → `8a32f24873ad8f28c8872e6a5eab7f0825c88812` (`perf-2026-03-14-88-g8a32f24873`)

Reproduce Arena: `git checkout dd6618c68 && git submodule update --init --recursive`.
Task `LMDrillLiftRlD1` (success = lift 0.15 + grasped) exists **only** on the `darrelldai/offline-rl-new`
branch; it is not on `masoud/g1_factory` or `main`.

### ⚠ Reproducibility caveats (don't skip)

0. **UPDATE (2026-07-03):** the RLinf Arena integration has since been committed to local branch
   `jk/isaaclab-arena-gr00t` (`79d52345`, not yet pushed). **Consequence for `sync_rlinf.sh`:** it ships
   only the *working-tree delta* (`git ls-files -mo`), which is now empty, so it delivers NOTHING; and the
   node's base clone is upstream GitHub, which does not have `79d52345`. To put the Arena code on a truly
   fresh node you must either (a) use the baked eval image (its RLinf already contains it — see §10), or
   (b) push the branch to the fork and `git clone` + `git checkout jk/isaaclab-arena-gr00t` on the node.
1. **RLinf Arena integration (originally uncommitted)** on top of `f711061e`. The whole env package
   `rlinf/envs/isaaclab_arena/` and configs are *untracked*, plus these are *modified*:
   `rlinf/envs/__init__.py`, `rlinf/envs/action_utils.py`, `rlinf/models/embodiment/gr00t/simulation_io.py`,
   `.../embodiment_tags.py`, `.../gr00t_n1d7/{__init__.py,gr00t_action_model.py}`,
   `rlinf/runners/{embodied_runner.py,async_embodied_runner.py}`,
   `rlinf/workers/actor/fsdp_sac_policy_worker.py`, `requirements/install.sh`; untracked configs under
   `examples/embodiment/config/` and `evaluations/isaaclab_arena/`.
   → **Snapshot these** (commit to a branch, or `git stash create`/`git diff > arena.patch` + archive the
   untracked dirs) before relying on `f711061e`. The eval nodes use this working tree via `PYTHONPATH`.
2. **Isaac-GR00T** node state = fresh clone of `ab88b50` **plus** `osmo_patch_isaac_gr00t.py` (which
   reproduces exactly the 5 modified files: `examples/finetune.sh`, `gr00t/configs/data/data_config.py`,
   `gr00t/data/dataset/{factory.py,sharded_single_step_dataset.py}`, `gr00t/experiment/launch_finetune.py`).
   So Isaac-GR00T IS reproducible from commit + patch (no manual snapshot needed).
3. **RESOLVED (2026-07-04):** `arena_local_bringup/` was moved into `RLinf/arena_local_bringup/` and
   committed on branch `jk/isaaclab-arena-gr00t` (`8a45bf2c`) — the scripts + this runbook are now
   version-controlled (heavy artifacts/videos/logs + `.wandb.env` are gitignored). Still LOCAL-only until
   the branch is pushed (see #1). NOTE: because it now lives inside RLinf, the eval-image build's
   `COPY RLinf` would sweep the (gitignored but not docker-ignored) video dirs — a `.dockerignore` must
   exclude `RLinf/arena_local_bringup/{tunevis_videos,eval_videos,eval_frames,videos,newds_inspect}` before
   the next `docker build` (see §10).
4. gr00t `jk/offline-rl` is only needed for the `gear` CLI and the internal IQL-baseline reference, not for
   the BC train/eval path itself.

---

## 10. Eval environment image (baked — the eval node is NOT runbook-buildable)

The eval half needs IsaacLab-Arena + its 28 GB IsaacSim-4.6.5 `.venv` + RLinf + the eval driver. This is too
heavy to build on a node, so it is **baked into a Docker image**; a fresh eval node = a cluster/workflow
launched from that image (ckpts + backbone are read from Lustre at runtime, NOT baked).

**Image:** `nvcr.io/nvidian/juekunl-arena-eval:n1d7-drilllift`
- local Docker id `3b98a0667e27` (~131 GB), base `nvcr.io/nvidian/gear-n2-eval:py312-2026.05.07` (sha256 `3ea9ec1cfb1b`)
- manifest list sha256 `3b98a0667e2731...`
- **Where it lives:** (a) local Docker daemon on this dev box; (b) pushed to the nvcr registry (OSMO nodes pull from here); (c) repo-tarball fallback `arena_stack.tar` (32.5 GB) at `~/Work/arena_stack.tar` and `s3://GearCheckpoints/juekunl/arena_stack.tar` (the pre-image extract-on-node path).

**Built from** `arena_local_bringup/Dockerfile.arena-eval` — it `COPY`s the repos to the SAME absolute paths
they were built at (`/home/juekunl/Work/{IsaacLab-Arena,Isaac-GR00T,RLinf}` + `eval_bc_ckpt.py`), because the
`.venv` uses PEP-660 editable installs with absolute finders. Build context is `~/Work` (~32 GB):
```bash
cd /home/juekunl/Work
# The build context is ~/Work; its .dockerignore allowlists IsaacLab-Arena/Isaac-GR00T/RLinf and must
# exclude RLinf/arena_local_bringup/{tunevis_videos,eval_videos,eval_frames,videos,newds_inspect} or
# `COPY RLinf` bloats the image by ~400MB. A reference copy is kept in the repo at
# RLinf/arena_local_bringup/work.dockerignore -> copy it to ~/Work/.dockerignore before building.
cp RLinf/arena_local_bringup/work.dockerignore .dockerignore
docker build -f RLinf/arena_local_bringup/Dockerfile.arena-eval -t nvcr.io/nvidian/juekunl-arena-eval:n1d7-drilllift .
docker push nvcr.io/nvidian/juekunl-arena-eval:n1d7-drilllift        # so OSMO can pull it
```

**Deploy to a cluster (pick one):**
- **OSMO workflow** referencing the image (self-contained, spins fresh pods): set `image:
  nvcr.io/nvidian/juekunl-arena-eval:n1d7-drilllift` in the workflow YAML (see `eval_sweep_osmo.yaml`, which
  runs the sweep straight from Lustre ckpts with zero on-node build), then `osmo workflow submit ...`.
- **gear ray cluster** launched from the image (gives an interactive node like `eval_ray_cluster_1n_...`),
  then drive it with `osmo_exec` per §7.
- The baked RLinf is frozen at image-build time; to run newer uncommitted RLinf on it, `sync_rlinf.sh`
  refreshes the code (§4). IsaacLab-Arena is pinned to `dd6618c68` inside the image.

**Reuse policy:** since the image already exists (local + nvcr), reuse it — either keep an existing
`eval_ray_cluster_1n_*` alive, or launch a new workflow/cluster from the image. Do NOT try to rebuild the
Arena venv on a bare node.
```
