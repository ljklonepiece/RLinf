"""S4.5 diagnostic — RLinf-PATH open-loop action reconstruction (is my eval plumbing right?).

Feeds the SAME drill_lift_rl demo frames through the RLinf ``gr00t_n1d7`` loader +
``convert_isaaclab_arena_g1_obs_to_gr00t_format`` + ``convert_to_isaaclab_arena_g1_action_n1d7``
(exactly the path eval_bc_ckpt.py uses at inference), and compares the predicted 35-d action
chunk to the dataset ground-truth ``action``. If MSE ~= the native reconstruction
(diag_recon_native.py), the RLinf obs converter + action decode are correct and the 0% sim
success is NOT in the prediction path.

Run (OSMO node, Arena 4.6.5 venv, needs 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 python diag_recon_rlinf.py \
    /mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000 \
    /mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl \
    "Lift the power drill from the table."
Result seen (ckpt-10000, 5 trajs): avg MSE 0.0049, avg MAE 0.018 (~native) -> converter + decode CORRECT.
"""

import sys

import numpy as np
import torch
import decord
import pyarrow.parquet as pq
from omegaconf import OmegaConf

CKPT = sys.argv[1]
D = sys.argv[2]
TASK = sys.argv[3] if len(sys.argv) > 3 else "Lift the power drill from the table."
BACKBONE = "/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B"

# Pure-BC head (no DSRL, no value head) -- matches eval_bc_ckpt.py.
RL_HEAD = {
    "joint_logprob": False, "noise_method": "flow_sde", "ignore_last": False,
    "safe_get_logprob": False, "noise_anneal": False, "noise_params": [0.7, 0.3, 400],
    "noise_level": 0.3, "add_value_head": False, "chunk_critic_input": False,
    "detach_critic_input": True, "disable_dropout": True, "use_vlm_value": False,
    "value_vlm_mode": "mean_token", "padding_value": 570, "use_dsrl": False,
}


def load_model():
    from rlinf.models.embodiment.gr00t.gr00t_n1d7 import get_model

    cfg = OmegaConf.create({
        "model_path": CKPT, "backbone_model_path": BACKBONE,
        "embodiment_tag": "isaaclab_arena_g1", "obs_converter_type": "isaaclab_arena_g1",
        "action_dim": 35, "num_action_chunks": 8, "denoising_steps": 4,
        "precision": "bf16", "rl_head_config": RL_HEAD,
    })
    m = get_model(cfg, torch_dtype=torch.bfloat16)
    m.to("cuda")
    m.eval()
    return m


def main():
    model = load_model()
    CH = 8
    all_se, all_ae = [], []
    for ep in [0, 1, 2, 3, 4]:
        ch = ep // 1000
        t = pq.read_table(f"{D}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")
        state = np.stack(t["observation.state"].to_pylist()).astype(np.float32)
        action = np.stack(t["action"].to_pylist()).astype(np.float32)
        vr = decord.VideoReader(
            f"{D}/videos/chunk-{ch:03d}/observation.images.ego_view/episode_{ep:06d}.mp4"
        )
        L = min(200, len(state), len(vr))
        preds = np.zeros((L, 35), np.float32)
        filled = np.zeros(L, bool)
        for t0 in range(0, L, CH):
            frame = vr[t0].asnumpy()
            env_obs = {
                "states": torch.from_numpy(state[t0:t0 + 1]),
                "main_images": torch.from_numpy(frame[None]),
                "task_descriptions": [TASK],
            }
            with torch.no_grad():
                ac, _ = model.predict_action_batch(env_obs, mode="eval")
            ac = ac.detach().cpu().numpy() if isinstance(ac, torch.Tensor) else np.asarray(ac)
            n = min(CH, L - t0, ac.shape[1])
            preds[t0:t0 + n] = ac[0, :n]
            filled[t0:t0 + n] = True
        m = filled
        se = float(((preds[m] - action[:L][m]) ** 2).mean())
        ae = float(np.abs(preds[m] - action[:L][m]).mean())
        all_se.append(se)
        all_ae.append(ae)
        print(f"traj {ep}: MSE={se:.6f} MAE={ae:.6f}", flush=True)
    print(f"AVG_RLINF MSE={np.mean(all_se):.6f} MAE={np.mean(all_ae):.6f} TASK={TASK!r}", flush=True)


if __name__ == "__main__":
    main()
