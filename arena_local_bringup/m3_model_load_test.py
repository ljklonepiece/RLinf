"""M3 model-leg proof: load the PUBLIC GR00T N1.7-3B via RLinf's own loader in the
single Arena venv (no Isaac Sim), to prove the 3B model + Cosmos-Reason2-2B backbone +
RL heads load and run a GPU forward alongside the sim deps (torch 2.10 / numpy 2.x).

This is the embodiment-INDEPENDENT architecture check. Policy quality is irrelevant;
we only care that get_model() returns a CUDA model. We try the intended Arena G1 tag
first; if the public base lacks its stats, we print the precise blocker.

Run:
  docker exec isaaclab-jk-mvp bash -lc \
    'cd /home/juekunl/Work/IsaacLab-Arena && source .venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=0 python /home/juekunl/Work/arena_local_bringup/m3_model_load_test.py'
"""

import traceback

import torch
from omegaconf import OmegaConf

MODEL = "/home/juekunl/Work/models/GR00T-N1.7-3B"
BACKBONE = "/home/juekunl/Work/models/Cosmos-Reason2-2B"

RL_HEAD = {
    "joint_logprob": False,
    "noise_method": "flow_sde",
    "ignore_last": False,
    "safe_get_logprob": False,
    "noise_anneal": False,
    "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.3,
    "add_value_head": True,
    "chunk_critic_input": False,
    "detach_critic_input": True,
    "disable_dropout": True,
    "use_vlm_value": False,
    "value_vlm_mode": "mean_token",
    "padding_value": 570,
    "use_dsrl": False,
}


def try_load(embodiment_tag: str, obs_converter_type: str = "isaaclab_arena_g1"):
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = OmegaConf.create(
        {
            "model_path": MODEL,
            "backbone_model_path": BACKBONE,
            "embodiment_tag": embodiment_tag,
            "obs_converter_type": obs_converter_type,
            "action_dim": 35,
            "num_action_chunks": 8,
            "denoising_steps": 4,
            "precision": "bf16",
            "rl_head_config": RL_HEAD,
        }
    )
    print(f"\n=== get_model(embodiment_tag={embodiment_tag!r}) ===", flush=True)
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  LOADED OK: {type(model).__name__}, params={n_params/1e9:.2f}B, "
          f"cuda_mem={mem:.1f}GB, dtype=bf16", flush=True)
    return model


def try_forward(model):
    """Run a real action prediction on dummy Arena-G1 obs (RLinf-canonical format)."""
    env_obs = {
        "states": torch.randn(1, 43),  # G1 43-d joint state
        "main_images": torch.randint(0, 255, (1, 224, 224, 3), dtype=torch.uint8),
        "task_descriptions": ["Lift the drill from the table."],
    }
    print("\n=== predict_action_batch(dummy Arena-G1 obs) ===", flush=True)
    with torch.no_grad():
        raw_action, result = model.predict_action_batch(env_obs, mode="eval")
    import numpy as np
    arr = np.asarray(raw_action)
    print(f"  FORWARD OK: action shape={arr.shape} (expect trailing dim 35), "
          f"result keys={list(result.keys())[:6]}", flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(), flush=True)
    # PLUMBING path: the public base only ships the real_g1_relative_eef G1 embodiment, so
    # use the eef converter + tag to drive a real forward (garbage policy, proves the loop).
    try:
        model = try_load("isaaclab_arena_g1_eef", "isaaclab_arena_g1_eef")
        print("\nMODEL_LOAD_OK: public N1.7 (3B + Cosmos backbone, sdpa) loaded on GPU "
              "in the single Arena venv.", flush=True)
        try:
            try_forward(model)
            print("\nMODEL_LEG_OK: load + Arena-G1(eef) forward both succeeded.", flush=True)
        except Exception:
            print("\n--- forward FAILED (load OK; obs/modality adaptation needed) ---", flush=True)
            traceback.print_exc()
    except Exception:
        print("\n--- LOAD FAILED ---", flush=True)
        traceback.print_exc()
