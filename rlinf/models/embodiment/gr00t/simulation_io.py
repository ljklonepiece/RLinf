# Copyright 2026 The RLinf Authors.
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

import numpy as np
import torch
import torch.nn.functional as F


def convert_libero_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}

    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.image"] = env_obs["main_images"].unsqueeze(1).numpy()
    groot_obs["video.wrist_image"] = env_obs["wrist_images"].unsqueeze(1).numpy()
    # [B, 8] -> [B, T(1), 8]
    groot_obs["state.x"] = env_obs["states"].unsqueeze(1)[:, :, 0:1].numpy()
    groot_obs["state.y"] = env_obs["states"].unsqueeze(1)[:, :, 1:2].numpy()
    groot_obs["state.z"] = env_obs["states"].unsqueeze(1)[:, :, 2:3].numpy()
    groot_obs["state.roll"] = env_obs["states"].unsqueeze(1)[:, :, 3:4].numpy()
    groot_obs["state.pitch"] = env_obs["states"].unsqueeze(1)[:, :, 4:5].numpy()
    groot_obs["state.yaw"] = env_obs["states"].unsqueeze(1)[:, :, 5:6].numpy()
    groot_obs["state.gripper"] = env_obs["states"].unsqueeze(1)[:, :, 6:].numpy()
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]

    return groot_obs


def convert_maniskill_obs_to_gr00t_format(env_obs):
    """
    Convert the observation to the format expected by GR00T models.
    The data format is determined by the modality_config and meta/info.json
    following LeRobot format.
    """
    groot_obs = {}
    # video
    # TODO(lx): If we have a dataset on maniskill, resize can be avoided.
    # But now we have to resize images to libero data version.
    env_obs["main_images"] = cut_and_resize_images(
        env_obs["main_images"],
        env_obs["main_images"].shape[-3],  # H
        256,
    )
    # [B, H, W, C] -> [B, T, H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).numpy()
    # state
    if "state" in env_obs:
        raise NotImplementedError("State from simulation are not unified yet.")
    else:
        # gr00t_1_7 pad the state to input dimension
        # create state of [B, T, C]
        groot_obs["state.left_arm"] = np.zeros((env_obs["main_images"].shape[0], 1, 7))
    # annotation
    groot_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_libero_action_n1d5(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T N1.5 action chunk to Libero format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array = normalize_gripper_action(action_array, binarize=True)
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d6(
    action_chunk: dict[str, np.array],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.6 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")

    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


def convert_to_libero_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Convert GR00T N1.7 action chunk to a 7-dim Libero action array.

    Gripper normalization is NOT applied here; it is handled by the shared
    ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.
    """
    try:
        pos = action_chunk["end_effector_position"][:, :chunk_size]
        rot = action_chunk["end_effector_rotation"][:, :chunk_size]
        gripper = action_chunk["gripper_close"][:, :chunk_size]
        action_array = np.concatenate([pos, rot, gripper], axis=-1)
    except KeyError:
        if all(
            key in action_chunk
            for key in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
        ):
            action_array = np.concatenate(
                [
                    action_chunk["x"][:, :chunk_size],
                    action_chunk["y"][:, :chunk_size],
                    action_chunk["z"][:, :chunk_size],
                    action_chunk["roll"][:, :chunk_size],
                    action_chunk["pitch"][:, :chunk_size],
                    action_chunk["yaw"][:, :chunk_size],
                    action_chunk["gripper"][:, :chunk_size],
                ],
                axis=-1,
            )
        elif "rel_arm_action" in action_chunk:
            arm = action_chunk["rel_arm_action"][:, :chunk_size]
            grp = action_chunk["gripper_action"][:, :chunk_size]
            action_array = np.concatenate([arm, grp], axis=-1)
        else:
            raise KeyError(f"can not find Action Keys: {list(action_chunk.keys())}")
    # gripper conversion is handled by the shared
    # ``prepare_actions_for_libero`` in ``rlinf.envs.action_utils``.

    if action_array.shape[-1] != 7:
        raise ValueError(f"Expected 7-dim action, got {action_array.shape[-1]}")
    return action_array


# ===== IsaacLab-Arena G1 (embodiment "g1", GR00T tag "isaaclab_arena_g1") =====
# Authoritative schema: IsaacLab-Arena g1 modality.json + gr00t MODALITY_CONFIGS
# ["isaaclab_arena_g1"]. State is a flat 43-d joint vector split into 7 named groups;
# action is 35-d in a DIFFERENT group order (no legs; adds base/navigate commands).

# State group -> slice in the canonical 43-d state. Order/slices match modality.json and
# g1_layout.G1_STATE_KEY_ORDER, so split(assemble(...)) is lossless. Wrist pose /
# eef_pose is intentionally omitted (not in the G1 inference state).
_G1_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}

# Action group concat order (modality.json "action" section / G1Gr00tAction layout) -> 35-d.
# NOTE this differs from the state order: arms+hands first, then waist + commands.
_G1_ACTION_KEY_ORDER = (
    "left_arm",  # 7
    "right_arm",  # 7
    "left_hand",  # 7
    "right_hand",  # 7
    "waist",  # 3
    "base_height_command",  # 1
    "navigate_command",  # 3
)  # total = 35


def convert_isaaclab_arena_g1_obs_to_gr00t_format(env_obs):
    """Convert canonical RLinf G1 obs to the GR00T N1.7 ``isaaclab_arena_g1`` input dict.

    Mirrors gr00t's ``IsaacLabArenaEnvWrapper``: the flat 43-d joint state is pre-split
    into the 7 named ``state.*`` groups (modality.json slices), the single ego camera
    becomes ``video.ego_view`` with a unit time axis, and the task string is placed under
    ``annotation.human.task_description`` (note: no ``.action`` infix, unlike libero).
    Wrist-pose / eef_pose state is omitted on purpose (not consumed at G1 inference).
    """
    states = env_obs["states"]
    if states.dim() == 2:  # [B, 43] -> [B, T(1), 43]
        states = states.unsqueeze(1)

    groot_obs = {}
    for group, (start, end) in _G1_STATE_SLICES.items():
        groot_obs[f"state.{group}"] = states[:, :, start:end].cpu().numpy()
    # [B, H, W, C] -> [B, T(1), H, W, C]
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).cpu().numpy()
    groot_obs["annotation.human.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_isaaclab_arena_g1_action_n1d7(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Concatenate GR00T N1.7 G1 action groups into the 35-d env action.

    The decode step returns keys with an ``action.`` prefix (e.g. ``action.left_arm``);
    the unprefixed form is accepted too. Concatenation order matches modality.json /
    Arena ``G1Gr00tAction`` (left_arm, right_arm, left_hand, right_hand, waist,
    base_height_command, navigate_command). navigate/base_height are NOT zeroed (matches
    gr00t eval); the arm groups are already absolute after the model's decode step.
    """
    parts = []
    for key in _G1_ACTION_KEY_ORDER:
        val = action_chunk.get(f"action.{key}")
        if val is None:
            val = action_chunk.get(key)
        if val is None:
            raise KeyError(
                f"Missing G1 action group '{key}'. "
                f"Available: {list(action_chunk.keys())}"
            )
        parts.append(np.asarray(val)[:, :chunk_size])
    action_array = np.concatenate(parts, axis=-1)
    if action_array.shape[-1] != 35:
        raise ValueError(f"Expected 35-dim G1 action, got {action_array.shape[-1]}")
    return action_array


# ===== IsaacLab-Arena G1 -> PUBLIC N1.7 base "real_g1_relative_eef_relative_joints" =====
# PLUMBING ONLY. The released public GR00T N1.7-3B ships modality + stats for
# ``real_g1_relative_eef_relative_joints`` (eef-based) but NOT the real Arena
# ``unitree_g1_full_body_with_waist_height_nav_cmd`` embodiment. To exercise the full
# RLinf<->Arena<->GR00T loop on the *public* base (sign-of-life; the policy is untrained
# for drill_lift), best-effort map the Arena 43-d joint state into this embodiment's
# groups: arms/hands/waist copy across from the canonical slices, the two wrist-eef groups
# are zero-filled (Arena's inference state has no eef pose), and the legs are dropped.
# A REAL drill_lift policy must instead use a ckpt finetuned on the Arena G1 embodiment
# (obs_converter_type="isaaclab_arena_g1").

# state group order/dims for the public base's real_g1_relative_eef embodiment (49-d).
_G1_EEF_STATE_GROUPS = (
    ("left_wrist_eef_9d", 9),
    ("right_wrist_eef_9d", 9),
    ("left_hand", 7),
    ("right_hand", 7),
    ("left_arm", 7),
    ("right_arm", 7),
    ("waist", 3),
)
# Arena canonical 43-d state slices for the groups we can fill (eef groups identity-filled).
_G1_EEF_SRC_SLICES = {
    "left_arm": (15, 22),
    "right_arm": (29, 36),
    "left_hand": (22, 29),
    "right_hand": (36, 43),
    "waist": (12, 15),
}
# Identity eef pose for the 9-d wrist groups: translation(3)=0 + rot6d(6)=identity (first
# two columns of I). A zero rot6d is a degenerate rotation matrix -> the model's
# relative->absolute action decode (Rotation.from_matrix SVD) fails to converge, so use I.
_G1_EEF_IDENTITY_9D = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def convert_isaaclab_arena_g1_eef_obs_to_gr00t_format(env_obs):
    """Best-effort map Arena G1 obs to the public base's real_g1_relative_eef state groups.

    PLUMBING ONLY (the public N1.7 base lacks the real Arena G1 modality). Wrist-eef groups
    are identity-pose filled; arms/hands/waist come from the Arena 43-d joint state; legs are
    dropped.
    """
    states = env_obs["states"]
    if states.dim() == 2:  # [B, 43] -> [B, T(1), 43]
        states = states.unsqueeze(1)
    batch = states.shape[0]

    groot_obs = {}
    for group, dim in _G1_EEF_STATE_GROUPS:
        if group in _G1_EEF_SRC_SLICES:
            start, end = _G1_EEF_SRC_SLICES[group]
            groot_obs[f"state.{group}"] = states[:, :, start:end].cpu().numpy()
        else:  # *_wrist_eef_9d: no eef pose in the Arena inference state -> identity pose
            groot_obs[f"state.{group}"] = np.broadcast_to(
                _G1_EEF_IDENTITY_9D, (batch, 1, dim)
            ).copy()
    groot_obs["video.ego_view"] = env_obs["main_images"].unsqueeze(1).cpu().numpy()
    groot_obs["annotation.human.task_description"] = env_obs["task_descriptions"]
    return groot_obs


def convert_to_isaaclab_arena_g1_eef_action(
    action_chunk: dict[str, np.ndarray],
    chunk_size: int = 1,
) -> np.ndarray:
    """Extract the 35-d Arena g1_gr00t action from the public base's 53-d eef action output.

    PLUMBING ONLY. Concatenation order matches Arena ``G1Gr00tAction`` (left_arm, right_arm,
    left_hand, right_hand, waist, base_height_command, navigate_command); the model's
    ``*_wrist_eef_9d`` action groups are dropped.
    """
    order = (
        "left_arm",  # 7
        "right_arm",  # 7
        "left_hand",  # 7
        "right_hand",  # 7
        "waist",  # 3
        "base_height_command",  # 1
        "navigate_command",  # 3
    )
    parts = []
    for key in order:
        val = action_chunk.get(f"action.{key}")
        if val is None:
            val = action_chunk.get(key)
        if val is None:
            raise KeyError(
                f"Missing G1-eef action group '{key}'. "
                f"Available: {list(action_chunk.keys())}"
            )
        parts.append(np.asarray(val)[:, :chunk_size])
    action_array = np.concatenate(parts, axis=-1)
    if action_array.shape[-1] != 35:
        raise ValueError(f"Expected 35-dim G1 action, got {action_array.shape[-1]}")
    return action_array


def convert_to_maniskill_action(
    action_chunk: dict[str, np.array], chunk_size: int = 16
) -> np.ndarray:
    """Convert GR00T action chunk to Maniskill format."""
    return action_chunk["action.left_arm"][:, :chunk_size]


def convert_to_isaaclab_stack_cube_action(
    action_chunk: dict[str, np.array], chunk_size: int = 1
) -> np.ndarray:
    """Convert GR00T action chunk to Isaaclab Stack Cube format."""
    action_components = [
        action_chunk["action.x"][:, :chunk_size],
        action_chunk["action.y"][:, :chunk_size],
        action_chunk["action.z"][:, :chunk_size],
        action_chunk["action.roll"][:, :chunk_size],
        action_chunk["action.pitch"][:, :chunk_size],
        action_chunk["action.yaw"][:, :chunk_size],
        action_chunk["action.gripper"][:, :chunk_size],
    ]
    action_array = np.concatenate(action_components, axis=-1)
    action_array[..., -1] = np.sign(action_array[..., -1])
    assert action_array.shape[-1] == 7, (
        f"Expected 7-dim action, got {action_array.shape[-1]}"
    )
    return action_array


OBS_CONVERSION = {
    "maniskill": convert_maniskill_obs_to_gr00t_format,
    "libero": convert_libero_obs_to_gr00t_format,
    "isaaclab_stack_cube": convert_libero_obs_to_gr00t_format,
    "isaaclab_arena_g1": convert_isaaclab_arena_g1_obs_to_gr00t_format,
    "isaaclab_arena_g1_eef": convert_isaaclab_arena_g1_eef_obs_to_gr00t_format,
}

ACTION_CONVERSION_N1D5 = {
    "libero": convert_to_libero_action_n1d5,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D6 = {
    "libero": convert_to_libero_action_n1d6,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
}

ACTION_CONVERSION_N1D7 = {
    "libero": convert_to_libero_action_n1d7,
    "maniskill": convert_to_maniskill_action,
    "isaaclab_stack_cube": convert_to_isaaclab_stack_cube_action,
    "isaaclab_arena_g1": convert_to_isaaclab_arena_g1_action_n1d7,
    "isaaclab_arena_g1_eef": convert_to_isaaclab_arena_g1_eef_action,
}

# Back-compat alias: the pre-split (N1.5-era) IsaacLab<->RLinf bridge extension
# (isaaclab_contrib.rl.rlinf.extension on IsaacLab 4.6.5) references the un-suffixed
# ``simulation_io.ACTION_CONVERSION``. Keep it pointing at the N1.5 map so the bridge's
# register() completes (and _patch_gr00t_get_model runs) instead of raising AttributeError.
ACTION_CONVERSION = ACTION_CONVERSION_N1D5


def cut_and_resize_images(
    images: torch.Tensor, crop_size: int, target_size: int = 256
) -> torch.Tensor:
    """Cut and resize the images to the crop size."""
    images_nchw = images.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    original_width = images_nchw.shape[-1]  # W
    start = (original_width - crop_size) // 2
    end = start + crop_size

    # Crop: keep batch, channels, full height; crop width to [start:end]
    cropped_tensor = images_nchw[:, :, :, start:end]  # [B, C, H, crop_W]

    # Resize: interpolate to target_size x target_size
    resized_tensor = F.interpolate(
        cropped_tensor,
        size=(target_size, target_size),
        mode="bilinear",  # Or 'bicubic' for smoother results
        align_corners=False,
    )  # [B, C, target_size, target_size]

    # Convert back to NHWC
    resized_nhwc = resized_tensor.permute(
        0, 2, 3, 1
    ).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    return resized_nhwc


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [+1,-1].
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action
