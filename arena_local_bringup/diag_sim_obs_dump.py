"""S4.5 diagnostic — dump the SIM reset obs vs the dataset (state + image format/values).

Builds the real Arena LMDrillLiftRlD1 env through RLinf (IsaaclabArenaG1InProcessEnv), resets,
and prints the state (43-d) + camera image (dtype/range/mean) that the model actually receives
at inference, next to the dataset's observation.state[0] and ego_view frame 0. Confirms whether
the sim obs the model sees matches the training distribution (state convention + image scale).

MUST keep the ``if __name__ == "__main__":`` guard: SubProcIsaacLabEnv uses the spawn start
method and re-imports the module in the child (a top-level env build would recurse -> RuntimeError).

Run (OSMO node, Arena 4.6.5 venv, needs 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 python diag_sim_obs_dump.py
Result seen: SIM_STATE first12 == DATA_STATE first12 (identical); SIM_IMAGE uint8 mean 108.0
vs DATA_IMAGE uint8 mean 107.2 -> sim obs (state + image) is CORRECT.
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    import numpy as np
    import torch  # noqa: F401
    from omegaconf import OmegaConf
    from rlinf.envs.isaaclab_arena import REGISTER_ISAACLAB_ARENA_ENVS

    cfg = OmegaConf.create({
        "seed": 42, "auto_reset": False, "use_rel_reward": False, "ignore_terminations": False,
        "reward_coef": 1.0, "max_episode_steps": 720,
        "video_cfg": {"save_video": False, "info_on_video": False, "fps": 20, "video_base_dir": "/tmp"},
        "init_params": {
            "id": "LMDrillLiftRlD1", "num_envs": 1, "device": "cuda:0",
            "task_description": "Lift the drill from the table.", "embodiment": "g1_gr00t",
            "lift_height": 0.15, "state_key": "robot_joint_pos", "camera_key": "robot_head_cam_rgb",
        },
    })
    env = REGISTER_ISAACLAB_ARENA_ENVS[cfg.init_params.id](
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )
    obs, _ = env.reset()
    print("OBS_KEYS", list(obs.keys()), flush=True)
    s = obs["states"]
    print("SIM_STATE", tuple(s.shape), s.dtype, "first12",
          [round(float(x), 3) for x in s.flatten()[:12]], flush=True)
    if "main_images" in obs:
        img = obs["main_images"]
        print("SIM_IMAGE", tuple(img.shape), img.dtype, "min", round(float(img.min()), 4),
              "max", round(float(img.max()), 4), "mean", round(float(img.float().mean()), 4), flush=True)
    else:
        print("SIM_IMAGE MISSING", flush=True)

    import pyarrow.parquet as pq
    D = "/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl"
    t = pq.read_table(f"{D}/data/chunk-000/episode_000000.parquet", columns=["observation.state"])
    ds0 = np.asarray(t["observation.state"].to_pylist()[0], dtype=float)
    print("DATA_STATE first12", [round(float(x), 3) for x in ds0[:12]], flush=True)
    import decord
    vr = decord.VideoReader(f"{D}/videos/chunk-000/observation.images.ego_view/episode_000000.mp4")
    f0 = vr[0].asnumpy()
    print("DATA_IMAGE", f0.shape, f0.dtype, "min", int(f0.min()), "max", int(f0.max()),
          "mean", round(float(f0.mean()), 4), flush=True)
    env.close()


if __name__ == "__main__":
    main()
