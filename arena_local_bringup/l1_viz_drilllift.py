"""L1 local bring-up: build IsaacLab-Arena G1 ``LMDrillLiftRlD1`` and step it.

This is the first "see it move" milestone for the GR00T-DSRL-on-RLinf plan. It
exercises ONLY the Arena env (no RLinf, no GR00T, no DSRL), launching Isaac Sim
locally so the drill-lift scene is visible, and verifies the raw observation /
action structure that the RLinf env wrapper will later have to map.

Run (GUI, watch it):
    cd /home/juekunl/Work/IsaacLab-Arena && source .venv/bin/activate
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python /home/juekunl/Work/arena_local_bringup/l1_viz_drilllift.py

Headless smoke (fast verification, no window):
    L1_HEADLESS=1 L1_STEPS=30 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python /home/juekunl/Work/arena_local_bringup/l1_viz_drilllift.py

Env vars:
    L1_HEADLESS=1   -> run without a window (default: GUI)
    L1_TASK=<name>  -> Arena env name (default: LMDrillLiftRlD1)
    L1_STEPS=<int>  -> number of env steps (default: 2000 GUI / set small for smoke)
    L1_ACTION=idle|zero|random  -> action to apply each step (default: idle)
"""

from __future__ import annotations

import os

# Isaac Sim blocks on an EULA stdin prompt on first import; auto-accept.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np
import torch

HEADLESS = os.environ.get("L1_HEADLESS", "0") == "1"
TASK = os.environ.get("L1_TASK", "LMDrillLiftRlD1")
NUM_STEPS = int(os.environ.get("L1_STEPS", "2000"))
ACTION_MODE = os.environ.get("L1_ACTION", "idle")
# Arena --embodiment selects the action space: g1_wbc_pink (default) -> 23-dim
# task-space WBC; g1 / g1_gr00t -> 35-dim joint-space groups (what the GR00T
# offline-RL worker uses); g1_wbc_joint -> 50-dim. Set L1_EMBODIMENT=g1 to mirror
# the RLinf/DSRL port. Empty = keep the env's default (23-dim).
EMBODIMENT = os.environ.get("L1_EMBODIMENT", "")

# Video recording: Omniverse-Kit's live on-screen window is unreliable over
# remote/virtual desktops (NoMachine/VNC), so the robust way to "watch" the task
# is to render the env's built-in third-person perspective camera to an MP4.
VIDEO = os.environ.get("L1_VIDEO", "1") == "1"
VIDEO_DIR = os.environ.get("L1_VIDEO_DIR", "/home/juekunl/Work/arena_local_bringup/videos")
VIDEO_EVERY = max(1, int(os.environ.get("L1_VIDEO_EVERY", "1")))  # capture 1 frame per N steps
VIDEO_FPS = int(os.environ.get("L1_VIDEO_FPS", "30"))

# g1_wbc_pink whole-body-controller idle command (keeps the robot stable).
# Layout (23): 2 hand + 3 L-wrist pos + 4 L-wrist quat(wxyz) + 3 R-wrist pos
#            + 4 R-wrist quat + 3 navigate + 1 base height + 3 torso rpy.
G1_WBC_PINK_IDLE_ACTION = [
    0.0, 0.0,
    0.201, 0.145, 0.101,
    1.000, 0.010, -0.008, -0.011,
    0.201, -0.145, 0.101,
    1.000, -0.010, -0.008, -0.011,
    0.0, 0.0, 0.0,
    0.75,
    0.0, 0.0, 0.0,
]


def _describe(obs, prefix: str = "") -> None:
    """Recursively print the structure of a (possibly nested) obs dict."""
    if isinstance(obs, dict):
        for key, value in obs.items():
            _describe(value, prefix=f"{prefix}{key}/")
    elif isinstance(obs, torch.Tensor):
        flat = int(np.prod(obs.shape[1:])) if obs.ndim > 1 else 0
        print(f"    {prefix:<40} shape={tuple(obs.shape)} dtype={obs.dtype} per_env_flat={flat}")
    else:
        print(f"    {prefix:<40} type={type(obs).__name__}")


def main() -> int:
    # Build the full Arena CLI parser (AppLauncher + IsaacLab + Arena + env subparsers).
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )

    parser = get_isaaclab_arena_environments_cli_parser()
    argv = ["--enable_cameras", "--num_envs", "1"]
    if HEADLESS:
        argv.append("--headless")
    else:
        # With --enable_cameras, IsaacLab defaults to the `isaaclab.python.rendering.kit`
        # experience, which only spawns a tiny 10x10 render window (invisible). Force the
        # full interactive experience so a real, watchable viewport window opens. Override
        # with L1_EXPERIENCE=... ; set L1_EXPERIENCE=0 to keep IsaacLab's default.
        experience = os.environ.get("L1_EXPERIENCE", "isaaclab.python.kit")
        if experience not in ("0", "", "default"):
            argv += ["--experience", experience]
    argv.append(TASK)  # positional subcommand must come last
    if EMBODIMENT:
        # --embodiment is an env-subcommand arg, so it must come AFTER the task.
        argv += ["--embodiment", EMBODIMENT]
    args_cli = parser.parse_args(argv)
    print(f"[L1] task={TASK} headless={HEADLESS} steps={NUM_STEPS} action={ACTION_MODE} device={args_cli.device}")

    # Start the simulation app BEFORE importing/building the env.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # The interactive `isaaclab.python.kit` experience (forced above for a visible
    # viewport) does NOT bake in `cameras_enabled = true` the way `rendering.kit` does,
    # so IsaacLab's camera sensor would abort with "spawned without --enable_cameras".
    # Set the gate flag here so RTX cameras render inside the interactive experience.
    if not HEADLESS:
        try:
            import carb

            carb.settings.get_settings().set_bool("/isaaclab/cameras_enabled", True)
            print("[L1] set /isaaclab/cameras_enabled=true for interactive experience")
        except Exception as exc:  # noqa: BLE001
            print(f"[L1] WARN: could not set cameras_enabled: {exc}")

    # Render stability: the RTX path can crash during the first render when DLSS is
    # asked to upscale a render product below its 300px minimum (seen as
    # "DLSS ... below minimal input resolution of 300" right before a silent exit).
    # Switch anti-aliasing off DLSS (0=off, 1=TAA, 2=FXAA, 3=DLSS, 4=DLAA).
    try:
        import carb

        carb.settings.get_settings().set("/rtx/post/aa/op", int(os.environ.get("L1_AA_OP", "2")))
        print(f"[L1] set /rtx/post/aa/op={os.environ.get('L1_AA_OP', '2')} (DLSS disabled)")
    except Exception as exc:  # noqa: BLE001
        print(f"[L1] WARN: could not set AA op: {exc}")

    # In non-headless mode IsaacLab's `--enable_cameras` selects the
    # `isaaclab.python.rendering.kit` experience, which spawns a tiny 10x10
    # "OpenGL Window" that is effectively invisible. Resize the Kit app window to
    # a visible size so the 3D viewport is watchable. Controlled by L1_WINDOW
    # (e.g. "1600x900"); set L1_WINDOW=0 to skip.
    if not HEADLESS:
        win_spec = os.environ.get("L1_WINDOW", "1600x900")
        if win_spec not in ("0", "", "off"):
            try:
                w_str, h_str = win_spec.lower().split("x")
                win_w, win_h = int(w_str), int(h_str)
                import omni.appwindow

                app_window = omni.appwindow.get_default_app_window()
                app_window.resize(win_w, win_h)
                for _ in range(5):
                    simulation_app.update()
                print(f"[L1] resized Isaac Sim window to {win_w}x{win_h} (set L1_WINDOW=0 to skip)")
            except Exception as exc:  # noqa: BLE001
                print(f"[L1] WARN: could not resize app window: {exc}")

    # Build the registered gym env.
    # render_mode="rgb_array" makes ManagerBasedRLEnv auto-instantiate its built-in
    # VideoRecorder. The recorder camera is taken from cfg.viewer.eye/lookat
    # (manager_based_rl_env.py), whose task default sits *in front of* the robot looking
    # at the drill, so the G1 itself is off-screen. Build the cfg, move the viewer eye
    # back to frame robot + table + drill, then make the env with that cfg.
    builder = get_arena_builder_from_cli(args_cli)
    render_mode = "rgb_array" if VIDEO else None
    if VIDEO:
        import gymnasium as gym

        from isaaclab_arena.environments.arena_env_builder import reapply_viewer_cfg

        def _vec3(name: str, default: str) -> tuple[float, float, float]:
            x, y, z = (float(v) for v in os.environ.get(name, default).split(","))
            return (x, y, z)

        cam_eye = _vec3("L1_CAM_EYE", "-2.5,3.0,2.2")
        cam_target = _vec3("L1_CAM_TARGET", "1.0,0.0,0.5")
        name, cfg = builder.build_registered()
        cfg.viewer.eye = cam_eye
        cfg.viewer.lookat = cam_target
        print(f"[L1] recorder camera eye={cam_eye} lookat={cam_target} (override via L1_CAM_EYE/L1_CAM_TARGET)")
        env = gym.make(name, cfg=cfg, render_mode=render_mode)
        reapply_viewer_cfg(env)
    else:
        env = builder.make_registered(render_mode=render_mode)

    obs, info = env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    print("\n[L1] ==== VERIFY: action space ====")
    print(f"    action_space = {env.action_space}")
    act_dim = int(np.prod(env.action_space.shape[1:])) if hasattr(env.action_space, "shape") else None
    print(f"    per-env action dim = {act_dim}")

    print("\n[L1] ==== VERIFY: observation structure (reset) ====")
    _describe(obs)

    # Choose an action tensor matching the env's action dim.
    if ACTION_MODE == "idle" and act_dim == len(G1_WBC_PINK_IDLE_ACTION):
        base = torch.tensor(G1_WBC_PINK_IDLE_ACTION, device=device, dtype=torch.float32)
        action = base.unsqueeze(0).repeat(num_envs, 1)
    elif ACTION_MODE == "random":
        action = torch.from_numpy(np.asarray(env.action_space.sample())).to(device).float()
    else:  # zero, or idle fell through on dim mismatch
        if ACTION_MODE == "idle" and act_dim != len(G1_WBC_PINK_IDLE_ACTION):
            print(f"[L1] WARN: idle action len {len(G1_WBC_PINK_IDLE_ACTION)} != act dim {act_dim}; using zeros")
        action = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)

    # Set up the MP4 writer (streaming, so we don't hold every frame in RAM).
    writer = None
    video_path = None
    n_frames = 0
    if VIDEO:
        import imageio.v2 as imageio

        os.makedirs(VIDEO_DIR, exist_ok=True)
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(VIDEO_DIR, f"{TASK}_{ACTION_MODE}_{stamp}.mp4")
        writer = imageio.get_writer(video_path, fps=VIDEO_FPS, macro_block_size=None)
        frame0 = env.render()
        if frame0 is None:
            print("[L1] WARN: env.render() returned None; video disabled")
            writer.close()
            writer = None
        else:
            writer.append_data(np.asarray(frame0))
            n_frames += 1
            print(f"[L1] recording video -> {video_path} (first frame {np.asarray(frame0).shape})")

    print(f"\n[L1] ==== stepping {NUM_STEPS} steps ====")
    n_term = 0
    for i in range(NUM_STEPS):
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(action)
        done = bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
        if done:
            n_term += 1
        if writer is not None and (i % VIDEO_EVERY == 0):
            frame = env.render()
            if frame is not None:
                writer.append_data(np.asarray(frame))
                n_frames += 1
        if i == 0:
            print("[L1] ==== VERIFY: first step return ====")
            print(f"    reward shape   = {tuple(torch.as_tensor(reward).shape)}")
            print(f"    terminated     = {torch.as_tensor(terminated).tolist()}")
            print(f"    truncated      = {torch.as_tensor(truncated).tolist()}")
        if (i + 1) % 200 == 0:
            print(f"    step {i + 1}/{NUM_STEPS}  reward={float(torch.as_tensor(reward).float().mean()):+.4f}  dones_so_far={n_term}")

    if writer is not None:
        writer.close()
        print(f"[L1] wrote {n_frames} frames -> {video_path}")

    print(f"\n[L1] DONE: ran {NUM_STEPS} steps, {n_term} episode end(s).")
    print("[L1] L1 OK")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
