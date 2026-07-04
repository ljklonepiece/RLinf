"""M0: build IsaacLab-Arena G1 ``LMDrillLiftRlD1`` on IsaacLab 7.5.0 and step it.

Goal: prove the Arena drill_lift task runs on the LATEST IsaacLab (ext 7.5.0),
the first gate of the "IsaacLab entrypoint -> RLinf -> Arena G1 env" path.

It deliberately BYPASSES ``isaaclab_arena_environments.cli`` (whose top-level
``ExampleEnvironments`` dict imports every example env, incl. dexsuite/kuka tasks
that are not yet ported to 7.5.0). Instead it builds the drill_lift env directly:
base Arena CLI parser + the ``LMDrillLiftRlD1`` subparser + ``ArenaEnvBuilder``.

Run (headless smoke, inside the gear container with the JK 7.5.0 venv active):
    L1_HEADLESS=1 L1_STEPS=10 python -u m0_drilllift_75.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np
import torch

HEADLESS = os.environ.get("L1_HEADLESS", "1") == "1"
NUM_STEPS = int(os.environ.get("L1_STEPS", "10"))
TASK = os.environ.get("L1_TASK", "LMDrillLiftRlD1")


def _describe(obs, prefix: str = "") -> None:
    if isinstance(obs, dict):
        for key, value in obs.items():
            _describe(value, prefix=f"{prefix}{key}/")
    elif isinstance(obs, torch.Tensor):
        flat = int(np.prod(obs.shape[1:])) if obs.ndim > 1 else 0
        print(f"    {prefix:<44} shape={tuple(obs.shape)} dtype={obs.dtype} per_env_flat={flat}")
    else:
        print(f"    {prefix:<44} type={type(obs).__name__}")


def main() -> int:
    # Base Arena parser (AppLauncher + Arena common args), NO env mega-import.
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena_environments.G1_Factory.LMDrillLift import LMDrillLiftRlD1

    env_cls = {"LMDrillLiftRlD1": LMDrillLiftRlD1}[TASK]

    parser = get_isaaclab_arena_cli_parser()
    sub = parser.add_subparsers(dest="example_environment", required=True)
    sp = sub.add_parser(env_cls.name)
    env_cls.add_cli_args(sp)

    argv = ["--enable_cameras", "--num_envs", "1"]
    if HEADLESS:
        argv.append("--headless")
    argv.append(env_cls.name)
    args_cli = parser.parse_args(argv)
    print(f"[M0] task={env_cls.name} headless={HEADLESS} steps={NUM_STEPS} device={getattr(args_cli, 'device', None)}")

    # Launch Isaac Sim BEFORE building the env.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Disable DLSS upscaling (crashes on tiny render products) for headless RTX.
    try:
        import carb

        carb.settings.get_settings().set("/rtx/post/aa/op", int(os.environ.get("L1_AA_OP", "2")))
    except Exception as exc:  # noqa: BLE001
        print(f"[M0] WARN: could not set AA op: {exc}")

    # Build the drill_lift env directly (mirrors get_arena_builder_from_cli without cli.py).
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    example_env = env_cls()
    builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    env = builder.make_registered(render_mode=None)

    obs, info = env.reset()
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    act_dim = int(np.prod(env.action_space.shape[1:])) if hasattr(env.action_space, "shape") else None

    print("\n[M0] ==== VERIFY: spaces ====")
    print(f"    action_space={env.action_space}  per-env action dim={act_dim}  num_envs={num_envs}  device={device}")
    print("\n[M0] ==== VERIFY: observation structure (reset) ====")
    _describe(obs)

    action = torch.zeros((num_envs, act_dim), device=device, dtype=torch.float32)
    print(f"\n[M0] ==== stepping {NUM_STEPS} steps (zero action) ====")
    n_term = 0
    reward = torch.zeros(num_envs)
    for i in range(NUM_STEPS):
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(action)
        if i == 0:
            print(f"    first step: reward_shape={tuple(torch.as_tensor(reward).shape)} "
                  f"terminated={torch.as_tensor(terminated).tolist()} truncated={torch.as_tensor(truncated).tolist()}")
        done = bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
        n_term += int(done)

    print(f"\n[M0] DONE: ran {NUM_STEPS} steps, {n_term} episode end(s), "
          f"last reward mean={float(torch.as_tensor(reward).float().mean()):+.4f}")
    print("[M0] M0 OK")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
