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

"""Canonical G1 observation layout for the IsaacLab-Arena suite.

This module is the *single source of truth* for how a raw Arena obs dict is turned
into RLinf's canonical obs (``main_images`` / ``states`` / ``task_descriptions``).
It is **numpy-only** (no torch, no rlinf, no isaaclab) so it imports cleanly in both
the Arena py3.12 server and the RLinf py3.11 worker, keeping the state-assembly order
identical on both sides of the socket.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Explicit assembly order for the flat 43-d G1 proprio state. The Arena ``policy``
# obs dict iterates as (left_arm, right_arm, left_hand, right_hand), but the GR00T
# base policy expects (left_arm, left_hand, right_arm, right_hand) per its
# ``G1_STATE_KEY_SLICES``. Assemble by this explicit order so the state matches the
# layout the policy was trained on; naive dict-order concatenation would swap groups.
G1_STATE_KEY_ORDER = (
    "left_leg",  # 6
    "right_leg",  # 6
    "waist",  # 3
    "left_arm",  # 7
    "left_hand",  # 7
    "right_arm",  # 7
    "right_hand",  # 7
)  # total = 43

# Arena camera key for the single G1 ego/head view.
G1_EGO_CAMERA_KEY = "robot_head_cam_rgb"


def to_numpy(x: Any) -> np.ndarray:
    """Convert a torch tensor (CUDA or CPU) or array-like to a CPU numpy array.

    Uses duck typing so this module never imports torch: any object exposing
    ``.detach()`` is treated as a tensor and moved to host memory.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):  # tensor without grad tracking
        return x.cpu().numpy()
    return np.asarray(x)


def assemble_state(policy_obs: dict) -> np.ndarray:
    """Concatenate the named G1 proprio groups into a flat (num_envs, 43) float32 state."""
    parts = []
    for key in G1_STATE_KEY_ORDER:
        if key not in policy_obs:
            raise KeyError(
                f"Arena policy obs missing expected state group '{key}'. "
                f"Available: {list(policy_obs.keys())}"
            )
        parts.append(to_numpy(policy_obs[key]))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def wrap_obs(obs: dict, task_description: str, num_envs: int) -> dict:
    """Map a raw Arena obs dict to RLinf's canonical obs dict (numpy payloads).

    Raw Arena obs groups (verified for embodiment ``g1``):
        policy/{left_leg,right_leg,waist,left_arm,right_arm,left_hand,right_hand}
        wbc/{robot_joint_pos,robot_joint_vel,left/right_wrist_pose_pelvis_frame}
        camera_obs/robot_head_cam_rgb  (num_envs, 480, 640, 3) uint8
        task_obs                       (num_envs, 1)
    """
    policy_obs = obs["policy"]
    camera_obs = obs["camera_obs"]

    states = assemble_state(policy_obs)
    main_images = to_numpy(camera_obs[G1_EGO_CAMERA_KEY])
    task_descriptions = [task_description] * num_envs

    return {
        "main_images": main_images,
        "states": states,
        "task_descriptions": task_descriptions,
    }
