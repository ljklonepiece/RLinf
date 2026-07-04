"""S4.5 diagnostic — NATIVE open-loop action reconstruction (is the BC ckpt good?).

Runs Isaac-GR00T's own ``gr00t.eval.open_loop_eval`` (Gr00tPolicy + LeRobotEpisodeLoader)
on the drill_lift_rl demos and reports the unnormalized action MSE/MAE vs ground truth.
This bypasses RLinf entirely: it tests whether the finetuned checkpoint reproduces the
demo actions under its OWN training-time transforms.

torchcodec is unavailable for torch 2.10, so we monkeypatch the loader's frame decode to
use ``decord`` (returns identical NHWC uint8 RGB frames).

Run (OSMO node, Arena 4.6.5 venv, needs 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 python diag_recon_native.py \
    /mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000 \
    /mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl \
    UNITREE_G1
Result seen (ckpt-10000, 5 trajs): avg MSE 0.0085, avg MAE 0.025 -> BC LEARNED the task.
"""

import sys

import numpy as np  # noqa: F401  (used indirectly by the loader)
import decord
import gr00t.data.dataset.lerobot_episode_loader as lel


def decord_frames(video_path, indices, decoder_kwargs=None):
    vr = decord.VideoReader(str(video_path))
    idx = [int(i) for i in indices]
    return vr.get_batch(idx).asnumpy()  # [N, H, W, C] uint8 RGB, matches torchcodec NHWC


lel.get_frames_by_indices = decord_frames

from gr00t.eval.open_loop_eval import ArgsConfig, main  # noqa: E402

if __name__ == "__main__":
    args = ArgsConfig(
        model_path=sys.argv[1],
        dataset_path=sys.argv[2],
        embodiment_tag=sys.argv[3] if len(sys.argv) > 3 else "UNITREE_G1",
        traj_ids=[0, 1, 2, 3, 4],
        steps=200,
        action_horizon=16,
    )
    main(args)
