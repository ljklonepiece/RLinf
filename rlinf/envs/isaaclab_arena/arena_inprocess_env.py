# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process IsaacLab-Arena G1 env for RLinf (no socket sidecar).

Used when RLinf runs in the SAME Python 3.12 venv as IsaacLab-Arena (the in-process
integration: IsaacLab entrypoint -> RLinf -> Arena env). Unlike the socket
``IsaaclabArenaG1Env``, this reuses :class:`IsaaclabBaseEnv`'s standard
``SubProcIsaacLabEnv`` machinery: ``_make_env_function`` returns a closure that, inside
the spawned child process, launches Isaac Sim via ``AppLauncher`` and builds the Arena
env directly through ``ArenaEnvBuilder`` -- mirroring ``get_arena_builder_from_cli`` but
importing only the requested env class, so it bypasses
``isaaclab_arena_environments.cli``'s import-everything ``ExampleEnvironments`` dict
(which pulls tasks, e.g. dexsuite, that may not be ported to the active IsaacLab).

Only env construction (``_make_env_function``) and obs conversion (``_wrap_obs``) are
overridden; metrics / step / chunk_step / reward come from ``IsaaclabBaseEnv``.
"""

from __future__ import annotations

import collections.abc

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

# Map an Arena env-class name (``init_params.id``) to ``module:class`` so the child can
# import just that class. Override per-task via ``init_params.arena_env_class``.
_ARENA_ENV_CLASSES = {
    "LMDrillLiftRlD1": "isaaclab_arena_environments.G1_Factory.LMDrillLift:LMDrillLiftRlD1",
    "LMDrillLiftRl": "isaaclab_arena_environments.G1_Factory.LMDrillLift:LMDrillLiftRl",
    "LMBoxLiftRlD1": "isaaclab_arena_environments.G1_Factory.LMBoxLift:LMBoxLiftRlD1",
}


class IsaaclabArenaG1InProcessEnv(IsaaclabBaseEnv):
    """In-process Arena G1 locomanip env (drill lift, box lift, ...) for RLinf."""

    def _make_env_function(self) -> collections.abc.Callable:
        """Return the child-process factory that builds the Arena env in-process.

        All IsaacLab/Arena imports happen inside the closure (after ``AppLauncher``),
        because it runs in the ``SubProcIsaacLabEnv`` spawned child.
        """
        task_id = self.isaaclab_env_id
        ip = self.cfg.init_params
        num_envs = self.num_envs
        embodiment = ip.get("embodiment", "g1_wbc_pink")
        lift_height = ip.get("lift_height", None)
        extra_args = dict(ip.get("arena_args", {}) or {})
        class_path = _ARENA_ENV_CLASSES.get(task_id) or ip.get("arena_env_class")
        if not class_path:
            raise ValueError(
                f"Arena task id '{task_id}' is not in _ARENA_ENV_CLASSES; set "
                "init_params.arena_env_class='module:Class' to build it in-process."
            )

        def make_env_arena() -> tuple:
            import importlib

            from isaaclab_arena.cli.isaaclab_arena_cli import (
                get_isaaclab_arena_cli_parser,
            )

            module_path, class_name = class_path.split(":", 1)
            env_cls = getattr(importlib.import_module(module_path), class_name)

            # Base Arena parser (AppLauncher + Arena common args), then attach ONLY this
            # task's subparser -- avoids importing the full ExampleEnvironments dict.
            parser = get_isaaclab_arena_cli_parser()
            sub = parser.add_subparsers(dest="example_environment", required=True)
            sp = sub.add_parser(env_cls.name)
            env_cls.add_cli_args(sp)

            argv = ["--enable_cameras", "--num_envs", str(num_envs), "--headless"]
            argv.append(env_cls.name)
            argv += ["--embodiment", str(embodiment)]
            if lift_height is not None:
                argv += ["--lift_height", str(lift_height)]
            for key, value in extra_args.items():
                argv += [f"--{key}", str(value)]
            args_cli = parser.parse_args(argv)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(args_cli).app

            # Disable DLSS upscaling (crashes on tiny render products) for headless RTX.
            try:
                import carb

                carb.settings.get_settings().set("/rtx/post/aa/op", 2)
            except Exception:  # noqa: BLE001
                pass

            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

            example_env = env_cls()
            builder = ArenaEnvBuilder(example_env.get_env(args_cli), args_cli)
            env = builder.make_registered(render_mode="rgb_array")
            return env.unwrapped, sim_app

        return make_env_arena

    # Canonical 43-d G1 state layout (matches simulation_io._G1_STATE_SLICES and the GR00T
    # modality.json group order), used to assemble a flat state from Arena's per-group obs.
    _G1_STATE_GROUP_ORDER = (
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "left_hand",
        "right_arm",
        "right_hand",
    )

    def _wrap_obs(self, obs: dict) -> dict:
        """Map the raw Arena obs groups to RLinf's canonical obs dict.

        Arena's ``g1_gr00t`` emits the G1 state pre-split into 7 named groups under the
        ``policy`` group (``left_leg``..``right_hand`` = 43-d), the flat 43-d vector under
        ``wbc/robot_joint_pos``, and the ego RGB camera under ``camera_obs``. We assemble a
        canonical flat 43-d ``states`` from the named policy groups when present (name-safe,
        order-correct), else fall back to a configured flat ``state_key`` (e.g. the
        ``g1_wbc_pink`` layout). Keys are configurable via ``init_params.state_key`` /
        ``camera_key``.
        """
        import torch

        ip = self.cfg.init_params
        state_key = ip.get("state_key", "robot_joint_pos")
        camera_key = ip.get("camera_key", "robot_head_cam_rgb")

        policy = obs.get("policy", obs) or {}
        wbc = obs.get("wbc", {}) or {}
        camera = obs.get("camera_obs", {}) or {}

        if isinstance(policy, dict) and all(
            g in policy for g in self._G1_STATE_GROUP_ORDER
        ):
            states = torch.cat(
                [policy[g] for g in self._G1_STATE_GROUP_ORDER], dim=-1
            )
        elif isinstance(policy, dict) and state_key in policy:
            states = policy[state_key]
        elif state_key in wbc:
            states = wbc[state_key]
        else:
            available = {
                "policy": list(policy) if isinstance(policy, dict) else type(policy),
                "wbc": list(wbc),
            }
            raise KeyError(
                f"Could not find G1 state (key '{state_key}' or the 7 named groups) in "
                f"Arena obs. Available groups: {available}"
            )

        env_obs = {
            "states": states.to(self.device),
            "task_descriptions": [self.task_description] * self.num_envs,
        }
        if camera_key in camera:
            env_obs["main_images"] = camera[camera_key].to(self.device)
        return env_obs
