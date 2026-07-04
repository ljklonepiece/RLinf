"""Dump the raw Arena obs structure for LMDrillLiftRlD1 + g1_gr00t embodiment, so we can
set the right state_key / camera_key in the in-process wrapper (g1_gr00t differs from the
g1_wbc_pink layout the earlier M2 test used).

Run:
  docker exec isaaclab-jk-mvp bash -lc \
    'cd /home/juekunl/Work/IsaacLab-Arena && source .venv/bin/activate && \
     CUDA_VISIBLE_DEVICES=0 python /home/juekunl/Work/arena_local_bringup/m3_dump_obs.py'
"""

import importlib

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

CLASS_PATH = "isaaclab_arena_environments.G1_Factory.LMDrillLift:LMDrillLiftRlD1"


def main():
    module_path, class_name = CLASS_PATH.split(":", 1)
    env_cls = getattr(importlib.import_module(module_path), class_name)

    parser = get_isaaclab_arena_cli_parser()
    sub = parser.add_subparsers(dest="example_environment", required=True)
    sp = sub.add_parser(env_cls.name)
    env_cls.add_cli_args(sp)
    argv = [
        "--enable_cameras", "--num_envs", "1", "--headless",
        env_cls.name, "--embodiment", "g1_gr00t", "--lift_height", "0.16",
    ]
    args_cli = parser.parse_args(argv)

    from isaaclab.app import AppLauncher

    sim_app = AppLauncher(args_cli).app

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    example_env = env_cls()
    builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
    env = builder.make_registered(render_mode="rgb_array").unwrapped

    obs, _ = env.reset()

    def dump(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                dump(v, f"{prefix}/{k}")
        else:
            shape = tuple(d.shape) if hasattr(d, "shape") else repr(type(d))
            print(f"  KEY {prefix}: {shape}")

    print("\n===== RAW ARENA OBS STRUCTURE (LMDrillLiftRlD1, g1_gr00t) =====", flush=True)
    dump(obs)
    print("===== END OBS STRUCTURE =====", flush=True)
    sim_app.close()


if __name__ == "__main__":
    main()
