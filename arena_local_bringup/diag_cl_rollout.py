"""S4.5 diagnostic — CLOSED-LOOP rollout characterization (BC drift vs action-application bug).

Runs the frozen BC policy closed-loop in the real Arena env and logs, per episode:
  - success + max shaped reward reached (how CLOSE to a grasp/lift the policy gets),
  - hand_act_absmax: magnitude the policy COMMANDS the hand-joint action dims (14:28),
  - l/rhand_state_dev: how much the hand joints ACTUALLY move (state 22:29 / 36:43).

Interpretation:
  - reaches + hands actuate + max_reward high but success=0  -> closed-loop DRIFT (plumbing OK; RL is the fix).
  - hands never move despite non-zero hand action                -> action-application bug (found the bug).

Same spawn/guard requirement as diag_sim_obs_dump.py. NOTE: launch in the FOREGROUND (a
nohup-backgrounded run gets killed during osmo session teardown before Isaac finishes booting).

Run (OSMO node, Arena 4.6.5 venv, needs 1 free GPU; ~80s env build + ~35s/episode):
  CUDA_VISIBLE_DEVICES=0 python diag_cl_rollout.py
STATUS: launched then interrupted (see EXPERIMENTS.md S4.5) -- re-run to finish the A/B call.
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("MUJOCO_GL", "egl")

CKPT = "/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000"
BACKBONE = "/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B"
RL_HEAD = {
    "joint_logprob": False, "noise_method": "flow_sde", "ignore_last": False,
    "safe_get_logprob": False, "noise_anneal": False, "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.3, "add_value_head": False, "chunk_critic_input": False,
    "detach_critic_input": True, "disable_dropout": True, "use_vlm_value": False,
    "value_vlm_mode": "mean_token", "padding_value": 570, "use_dsrl": False,
}


def main():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.isaaclab_arena import REGISTER_ISAACLAB_ARENA_ENVS
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    ecfg = OmegaConf.create({
        "seed": 42, "auto_reset": False, "use_rel_reward": False, "ignore_terminations": False,
        "reward_coef": 1.0, "max_episode_steps": 720,
        "video_cfg": {"save_video": False, "info_on_video": False, "fps": 20, "video_base_dir": "/tmp"},
        "init_params": {
            "id": "LMDrillLiftRlD1", "num_envs": 1, "device": "cuda:0",
            "task_description": "Lift the drill from the table.", "embodiment": "g1_gr00t",
            "lift_height": 0.15, "state_key": "robot_joint_pos", "camera_key": "robot_head_cam_rgb",
        },
    })
    env = REGISTER_ISAACLAB_ARENA_ENVS[ecfg.init_params.id](
        cfg=ecfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )
    mcfg = OmegaConf.create({
        "model_path": CKPT, "backbone_model_path": BACKBONE,
        "embodiment_tag": "isaaclab_arena_g1", "obs_converter_type": "isaaclab_arena_g1",
        "action_dim": 35, "num_action_chunks": 8, "denoising_steps": 4,
        "precision": "bf16", "rl_head_config": RL_HEAD,
    })
    model = get_model(mcfg, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    dev = env.device

    for ep in range(3):
        obs, _ = env.reset()
        steps = 0
        maxr = -1e9
        success = False
        hs0 = obs["states"][0, 22:29].cpu().numpy().copy()
        rhs0 = obs["states"][0, 36:43].cpu().numpy().copy()
        hs_dev = rhs_dev = hand_act_absmax = 0.0
        while steps < 720:
            with torch.no_grad():
                ac, _ = model.predict_action_batch(obs, mode="eval")
            ac = ac.detach().cpu().numpy() if isinstance(ac, torch.Tensor) else np.asarray(ac)
            hand_act_absmax = max(hand_act_absmax, float(np.abs(ac[0, :, 14:28]).max()))
            brk = False
            for i in range(min(8, ac.shape[1])):
                a = torch.as_tensor(ac[:, i]).to(dev).float()
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                maxr = max(maxr, float(np.asarray(r).flatten()[0]))
                cs = obs["states"][0, 22:29].cpu().numpy()
                crs = obs["states"][0, 36:43].cpu().numpy()
                hs_dev = max(hs_dev, float(np.abs(cs - hs0).max()))
                rhs_dev = max(rhs_dev, float(np.abs(crs - rhs0).max()))
                td = float(np.asarray(term).flatten()[0]) > 0.5
                tr = float(np.asarray(trunc).flatten()[0]) > 0.5
                if td or tr:
                    success = td
                    brk = True
                    break
            if brk:
                break
        print(f"EP{ep} success={int(success)} steps={steps} max_reward={maxr:.4f} "
              f"hand_act_absmax={hand_act_absmax:.4f} lhand_state_dev={hs_dev:.4f} "
              f"rhand_state_dev={rhs_dev:.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
