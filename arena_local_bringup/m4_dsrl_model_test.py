"""M4 DSRL model-leg probe: load the public N1.7 with use_dsrl=True (SAC steering heads) and
run predict_action_batch (-> sac_forward) on dummy Arena-G1(eef) obs. Fast, no Isaac Sim --
isolates DSRL-specific issues (noise dim, SAC heads) before the slow full train.py run.

Run:
  docker exec isaaclab-jk-mvp bash -lc \
    'cd /home/juekunl/Work/IsaacLab-Arena && source .venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=0 python /home/juekunl/Work/arena_local_bringup/m4_dsrl_model_test.py'
"""

import traceback

import numpy as np
import torch
from omegaconf import OmegaConf

MODEL = "/home/juekunl/Work/models/GR00T-N1.7-3B"
BACKBONE = "/home/juekunl/Work/models/Cosmos-Reason2-2B"

RL_HEAD = {
    # PPO/base fields (read by get_model + model __init__)
    "joint_logprob": False,
    "noise_method": "flow_sde",
    "ignore_last": False,
    "safe_get_logprob": False,
    "noise_anneal": False,
    "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.3,
    "add_value_head": False,  # DSRL trains its own SAC critic
    "chunk_critic_input": False,
    "detach_critic_input": True,
    "disable_dropout": True,
    "use_vlm_value": False,
    "value_vlm_mode": "mean_token",
    "padding_value": 570,
    # DSRL
    "use_dsrl": True,
    "dsrl_state_dim": 43,
    "dsrl_num_q_heads": 10,
    "dsrl_agg_q": "mean",
    "dsrl_image_latent_dim": 64,
    "dsrl_state_latent_dim": 64,
    "dsrl_hidden_dims": [128, 128, 128],
}


def main():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = OmegaConf.create(
        {
            "model_path": MODEL,
            "backbone_model_path": BACKBONE,
            "embodiment_tag": "isaaclab_arena_g1_eef",
            "obs_converter_type": "isaaclab_arena_g1_eef",
            "action_dim": 35,
            "num_action_chunks": 8,
            "denoising_steps": 4,
            "precision": "bf16",
            "rl_head_config": RL_HEAD,
        }
    )
    print("=== get_model(use_dsrl=True) ===", flush=True)
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    print(f"  DSRL model loaded: {type(model).__name__}, use_dsrl={getattr(model, 'use_dsrl', None)}", flush=True)

    env_obs = {
        "states": torch.randn(2, 43),
        "main_images": torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8),
        "task_descriptions": ["Lift the drill from the table."] * 2,
    }
    print("=== predict_action_batch (train mode -> sac_forward) ===", flush=True)
    with torch.no_grad():
        raw, result = model.predict_action_batch(env_obs, mode="train")
    print(f"  DSRL FORWARD OK: action={np.asarray(raw).shape}, result keys={list(result.keys())[:8]}", flush=True)
    print("M4_DSRL_MODEL_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n--- DSRL model probe FAILED ---", flush=True)
        traceback.print_exc()
