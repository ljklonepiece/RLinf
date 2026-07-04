"""Evaluate a PUBLIC GR00T N1.7 BC checkpoint on IsaacLab-Arena drill_lift, via RLinf's
public ``gr00t_n1d7`` loader (the correct arch for public Isaac-GR00T ckpts -- NOT the
internal omni ``gr00t``/``isaac_gr00t`` loaders).

Follows the internal gr00t eval methodology (groot/vla/eval/sim/main/run_eval_daemon.sh):
  n_episodes=100, seed=42, g1 35-d action, n_action_steps=8, max_episode_steps=720,
  terminate-on-success. Success is the Arena env's OWN terminal (lift 0.15 + grasped for
  ``*RlD1``), so numbers are comparable to the internal IQL baseline.

The eval env + success criterion are identical to the internal eval (same Arena
``LMDrillLiftRlD1``); only the policy LOADER differs (public gr00t_n1d7, matching the ckpt).

Run (gear container, Arena 4.6.5 venv):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -u eval_bc_ckpt.py \
    --model_path /home/juekunl/Work/bc_ckpts/checkpoint-2000/checkpoint-2000 \
    --n_episodes 100 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np
import torch
from omegaconf import OmegaConf

BACKBONE = "/home/juekunl/Work/models/Cosmos-Reason2-2B"

# Pure-BC head config: no DSRL, no value head (predict_action_batch uses the BC denoise path;
# the get_value guard makes prev_values zeros when there is no value head).
RL_HEAD = {
    "joint_logprob": False,
    "noise_method": "flow_sde",
    "ignore_last": False,
    "safe_get_logprob": False,
    "noise_anneal": False,
    "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.3,
    "add_value_head": False,
    "chunk_critic_input": False,
    "detach_critic_input": True,
    "disable_dropout": True,
    "use_vlm_value": False,
    "value_vlm_mode": "mean_token",
    "padding_value": 570,
    "use_dsrl": False,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_env(seed: int, max_episode_steps: int, video_dir: str | None = None):
    from rlinf.envs.isaaclab_arena import REGISTER_ISAACLAB_ARENA_ENVS

    cfg = OmegaConf.create(
        {
            "seed": seed,
            "auto_reset": False,
            "use_rel_reward": False,
            "ignore_terminations": False,  # respect success/timeout so success rate is real
            "reward_coef": 1.0,
            "max_episode_steps": max_episode_steps,
            "video_cfg": {
                "save_video": bool(video_dir),
                "info_on_video": True,  # overlay reward + termination on frames
                "fps": 20,
                "video_base_dir": video_dir or "/tmp",
            },
            "init_params": {
                "id": "LMDrillLiftRlD1",
                "num_envs": 1,
                "device": "cuda:0",
                "task_description": "Lift the drill from the table.",
                "embodiment": "g1_gr00t",  # 35-d GR00T action layout
                "lift_height": 0.15,  # *RlD1 success threshold (matches internal eval worker)
                "state_key": "robot_joint_pos",
                "camera_key": "robot_head_cam_rgb",
            },
        }
    )
    env_cls = REGISTER_ISAACLAB_ARENA_ENVS[cfg.init_params.id]
    return env_cls(cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)


def load_model(model_path: str, backbone: str):
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = OmegaConf.create(
        {
            "model_path": model_path,
            "backbone_model_path": backbone,
            "embodiment_tag": "isaaclab_arena_g1",  # ckpt was finetuned on the real G1 embodiment
            "obs_converter_type": "isaaclab_arena_g1",
            "action_dim": 35,
            "num_action_chunks": 8,
            "denoising_steps": 4,
            "precision": "bf16",
            "rl_head_config": RL_HEAD,
        }
    )
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    return model


def _first_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    return float(np.asarray(x).astype(float).flatten()[0])


def _done_and_success(term, trunc) -> tuple[bool, bool]:
    """Task success = the Arena success terminal fired (lift 0.15 + grasped).

    With ``ignore_terminations=False``, RLinf's env maps the Arena success terminal to
    ``terminations`` (term) and the timeout to ``truncations`` (trunc). So success = terminated
    (and NOT truncated). We must NOT use ``info['episode']['success_once']`` -- that is
    ``reward>0 ever``, which is True from step 1 under drill_lift's shaped/dense reward.
    Returns (episode_done, success).
    """
    terminated = _first_scalar(term) > 0.5
    truncated = _first_scalar(trunc) > 0.5
    return (terminated or truncated), terminated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_action_steps", type=int, default=8)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--video_dir", default=None,
                    help="Dir to save one MP4 per episode. If omitted, videos are saved BY DEFAULT "
                         "to ./eval_videos/<ckpt>/seed_<seed>; pass --no_video to disable.")
    ap.add_argument("--no_video", action="store_true", help="Disable video recording (on by default).")
    args = ap.parse_args()

    # Video recording is ON by default (auto path derived from ckpt + seed); --no_video opts out.
    if args.no_video:
        video_dir = None
    else:
        video_dir = args.video_dir or os.path.join(
            "eval_videos", os.path.basename(os.path.normpath(args.model_path)), f"seed_{args.seed}"
        )

    seed_everything(args.seed)
    print(f"[eval] model={args.model_path}\n[eval] n_episodes={args.n_episodes} seed={args.seed} "
          f"n_action_steps={args.n_action_steps} max_steps={args.max_episode_steps} "
          f"video_dir={video_dir}", flush=True)

    base_env = build_env(args.seed, args.max_episode_steps, video_dir)
    if video_dir:
        from rlinf.envs.wrappers.record_video import RecordVideo

        env = RecordVideo(base_env, base_env.video_cfg)
    else:
        env = base_env
    model = load_model(args.model_path, args.backbone)
    print(f"[eval] env + model ready (device={base_env.device}) video={'on' if video_dir else 'off'}", flush=True)

    successes = 0
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        success = False
        steps = 0
        while steps < args.max_episode_steps:
            with torch.no_grad():
                action_chunks, _ = model.predict_action_batch(obs, mode="eval")
            if isinstance(action_chunks, torch.Tensor):
                ac = action_chunks.detach().cpu()  # [B, num_chunks, 35]
            else:
                ac = torch.as_tensor(np.asarray(action_chunks))
            n = min(args.n_action_steps, ac.shape[1])
            episode_done = False
            for i in range(n):
                a = ac[:, i].to(base_env.device).float()
                obs, reward, term, trunc, infos = env.step(a)
                steps += 1
                done_now, s = _done_and_success(term, trunc)
                if done_now:
                    success = s
                    episode_done = True
                    break
                if steps >= args.max_episode_steps:
                    episode_done = True
                    break
            if episode_done:
                break
        successes += int(success)
        if video_dir:
            tag = "success" if success else "timeout"
            env.flush_video(video_sub_dir=f"ep{ep + 1:03d}_{tag}")
        print(f"  ep {ep + 1:3d}/{args.n_episodes}: success={int(success)} steps={steps} "
              f"(running {successes}/{ep + 1} = {successes / (ep + 1):.3f})", flush=True)

    rate = successes / args.n_episodes
    print(f"\nSUCCESS_RATE {args.model_path}: {successes}/{args.n_episodes} = {rate:.4f}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
