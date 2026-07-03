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

"""Default IsaacLab-Arena CLI arguments for the G1 locomanipulation suite.

These mirror the arguments the GR00T offline-RL eval worker passes when it builds
an Arena env (``isaaclab_arena_worker.py::_isaaclab_arena_cli_defaults``), so the
RLinf-hosted env is byte-for-byte compatible with the policy trained against it.

The key choice is ``embodiment="g1"`` (alias of ``g1_gr00t``), which selects the
35-dim joint-space action space (``G1Gr00tAction``: left_arm 7 + right_arm 7 +
left_hand 7 + right_hand 7 + waist 3 + base_height 1 + navigate 3). The Arena
default ``g1_wbc_pink`` would instead expose a 23-dim task-space WBC action, which
does not match the GR00T policy output.
"""

from __future__ import annotations

from typing import Any


def isaaclab_arena_cli_defaults(arena_env_name: str) -> dict[str, Any]:
    """Return the default Arena CLI namespace values for ``arena_env_name``.

    Args:
        arena_env_name: Arena env class name, e.g. ``"LMDrillLiftRlD1"``.

    Returns:
        A dict suitable for ``argparse.Namespace(**defaults)`` consumed by
        ``isaaclab_arena_environments.cli.get_arena_builder_from_cli``.
    """
    defaults: dict[str, Any] = {
        "num_envs": 1,
        "env_spacing": 2.5,
        "device": "cuda:0",
        "disable_fabric": False,
        "mimic": False,
        "solve_relations": False,
        "example_environment": arena_env_name,
        "enable_cameras": True,
        # 35-dim joint-space action that matches the GR00T policy (see module docstring).
        "embodiment": "g1",
        "teleop_device": None,
        "object": "power_drill",
        "background": "factory_room",
        "table": "white_table",
        "lift_height": 0.2,
        "episode_length_s": 50.0,
        "table_x": 2.0,
        "table_y": 0.0,
    }

    if arena_env_name.startswith("LMBoxLift"):
        defaults.update(
            {
                "object": "locomanip_cardbox",
                "table": "locomanip_white_table",
            }
        )
    if arena_env_name.endswith("RlD1"):
        defaults["lift_height"] = 0.15

    return defaults
