"""M2: prove RLinf's env wrapper builds + steps the Arena drill_lift env IN-PROCESS.

This instantiates ``IsaaclabArenaG1InProcessEnv`` directly (no Ray, no runner). The
wrapper's ``SubProcIsaacLabEnv`` spawns a child that launches Isaac Sim and builds
``LMDrillLiftRlD1`` via ``ArenaEnvBuilder``; the parent drives it over the standard
RLinf IPC. Proves the env half of: IsaacLab entrypoint -> RLinf -> Arena env.

Run (inside the gear container, Arena 4.6.5 venv active):
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u m2_inprocess_env_test.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import torch
from omegaconf import OmegaConf

ACTION_DIM = int(os.environ.get("M2_ACTION_DIM", "23"))  # g1_wbc_pink = 23
NUM_STEPS = int(os.environ.get("M2_STEPS", "5"))


def _describe(obs) -> None:
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key:<22} shape={tuple(value.shape)} dtype={value.dtype} dev={value.device}")
        else:
            print(f"    {key:<22} {type(value).__name__}={value if not isinstance(value, list) else value[:1]}")


def main() -> int:
    from rlinf.envs.isaaclab_arena import REGISTER_ISAACLAB_ARENA_ENVS

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "max_episode_steps": 256,
            "video_cfg": {"save_video": False, "info_on_video": False, "fps": 20, "video_base_dir": "/tmp"},
            "init_params": {
                "id": "LMDrillLiftRlD1",
                "num_envs": 1,
                "device": "cuda:0",
                "task_description": "Lift the drill from the table.",
                "embodiment": "g1_wbc_pink",
                "lift_height": 0.16,
                "state_key": "robot_joint_pos",
                "camera_key": "robot_head_cam_rgb",
            },
        }
    )

    env_cls = REGISTER_ISAACLAB_ARENA_ENVS[cfg.init_params.id]
    print(f"[M2] env class = {env_cls.__name__} for id={cfg.init_params.id}")
    print("[M2] constructing env (spawns child: Isaac Sim + ArenaEnvBuilder)...")
    env = env_cls(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    print(f"[M2] env built. device={env.device}")

    obs, info = env.reset()
    print("[M2] ==== VERIFY: reset obs (RLinf canonical) ====")
    _describe(obs)

    # g1_wbc_pink idle action: zeros give zero-norm wrist quats -> R.from_quat crash.
    # Layout (23): 2 hand | 3 L-wrist pos + 4 L-wrist quat(wxyz) | 3 R-wrist pos + 4 R-wrist
    # quat | 3 navigate | 1 base height | 3 torso rpy.
    g1_wbc_pink_idle = [
        0.0, 0.0,
        0.201, 0.145, 0.101, 1.000, 0.010, -0.008, -0.011,
        0.201, -0.145, 0.101, 1.000, -0.010, -0.008, -0.011,
        0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0,
    ]
    action = torch.tensor(g1_wbc_pink_idle, device=env.device, dtype=torch.float32).unsqueeze(0)
    print(f"\n[M2] ==== stepping {NUM_STEPS} steps (idle WBC-pink action, dim={action.shape[-1]}) ====")
    last = None
    for i in range(NUM_STEPS):
        obs, reward, term, trunc, infos = env.step(action)
        last = (reward, term, trunc)
        if i == 0:
            print(f"    first step: reward={torch.as_tensor(reward).flatten().tolist()} "
                  f"term={torch.as_tensor(term).flatten().tolist()} trunc={torch.as_tensor(trunc).flatten().tolist()}")
            print(f"    info.episode keys: {list(infos.get('episode', {}).keys())}")

    reward, term, trunc = last
    print(f"\n[M2] DONE: {NUM_STEPS} steps. last reward={float(torch.as_tensor(reward).float().mean()):+.4f}")
    print("[M2] M2 OK: RLinf in-process wrapper reset+stepped the Arena env")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
