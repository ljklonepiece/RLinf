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

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """The Fourier GR1 robot."""

    OXE_DROID = "oxe_droid"
    """The Open-X-Embodiment DROID robot with relative joint position actions."""

    AGIBOT_GENIE1 = "agibot_genie1"
    """The AgiBot Genie-1 with gripper dataset."""

    LIBERO_FRANKA = "libero_franka"
    """The Libero Franka dataset."""

    MANISKILL_WIDOWX = "maniskill_widowx"
    """The ManiSkill WidowX dataset."""

    ISAACLAB_FRANKA = "isaaclab_franka"
    """The Isaac Lab Franka dataset."""

    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """The RoboCasa Panda robot with omron mobile base."""

    UNITREE_G1 = "unitree_g1"
    """The Unitree G1 robot."""

    LIBERO_PANDA = "libero_panda"
    """The Libero Panda robot."""

    LIBERO_SIM = "libero_sim"
    """The Libero SIM robot for GR00T N1.7."""

    OXE_GOOGLE = "oxe_google"
    """The Open-X-Embodiment Google robot."""

    OXE_WIDOWX = "oxe_widowx"
    """The Open-X-Embodiment WidowX robot."""

    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    """The Behavior R1 Pro robot."""

    ISAACLAB_ARENA_G1 = "unitree_g1_full_body_with_waist_height_nav_cmd"
    """IsaacLab-Arena Unitree G1 (full body + waist + height + nav command, 35-d action).

    The *value* is the GR00T embodiment key that the public Isaac-GR00T ``UNITREE_G1``
    finetune writes into the checkpoint's modality config / ``dataset_statistics.json``,
    so the model's normalization lookup ``modality_configs[embodiment_tag.value]`` resolves.
    The RLinf-side obs/action format converters are keyed separately by ``obs_converter_type``
    (``"isaaclab_arena_g1"``), so they are unaffected by this value.
    """

    ISAACLAB_ARENA_G1_EEF = "real_g1_relative_eef_relative_joints"
    """PUBLIC GR00T N1.7 base's only shipped G1 embodiment (relative-eef + relative-joints).

    PLUMBING ONLY. The released ``nvidia/GR00T-N1.7-3B`` base ships modality/stats for this
    eef-based G1 (so ``modality_configs[value]`` resolves) but NOT the real Arena
    ``unitree_g1_full_body_with_waist_height_nav_cmd``. Pair with
    ``obs_converter_type="isaaclab_arena_g1_eef"`` to drive the full RLinf<->Arena<->GR00T
    loop on the *public* base for a sign-of-life run (the policy is untrained for drill_lift).
    A real drill_lift policy uses ``ISAACLAB_ARENA_G1`` + a finetuned ckpt instead.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """Any new embodiment used during post-training."""


# Embodiment tag string -> projector index in the Action Expert Module.
# Shared by GR00T N1.5 / N1.6 / N1.7 (N1.6-only tags are unused by N1.5 loaders).
# IDs must match official gr00t (see gr00t_n1d6 / gr00t_n1d7 processing / embodiment_id.json).
# GR00T N1.7 actually not use this mapping, because of new processor loading method.
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.LIBERO_PANDA.value: 2,
    EmbodimentTag.ROBOCASA_PANDA_OMRON.value: 13,
    EmbodimentTag.LIBERO_FRANKA.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.MANISKILL_WIDOWX.value: 30,
    EmbodimentTag.ISAACLAB_FRANKA.value: 31,
    EmbodimentTag.ISAACLAB_ARENA_G1.value: 25,
    EmbodimentTag.ISAACLAB_ARENA_G1_EEF.value: 25,
    EmbodimentTag.NEW_EMBODIMENT.value: 10,
}
