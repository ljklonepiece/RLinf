#!/usr/bin/env python3
"""S2.3 unit check: GR00T <-> IsaacLab-Arena G1 obs/action adapter (no 3B ckpt).

The only model<->env coupling for the Arena G1 task. Verifies the converters in
``rlinf/models/embodiment/gr00t/simulation_io.py`` and the registrations, against the
authoritative gr00t ``isaaclab_arena_g1`` modality schema:

  obs (env -> GR00T):  flat 43-d state -> 7 ``state.*`` groups (modality.json slices),
                       ego cam -> ``video.ego_view`` ([B,1,H,W,C]),
                       task -> ``annotation.human.task_description`` (no ``.action``).
  action (GR00T -> env): groups -> 35-d in order
                       [left_arm, right_arm, left_hand, right_hand, waist,
                        base_height_command, navigate_command].

Round-trips prove the split/concat ORDER matches the schema (assemble->split and
split->concat are identity). Also checks: converters registered under
``isaaclab_arena_g1``; ``EmbodimentTag.ISAACLAB_ARENA_G1`` exists; the env-side
``prepare_actions`` passes the 35-d action through unchanged.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s2_3_gr00t_arena_adapter_unit.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.models.embodiment.gr00t import simulation_io as sio  # noqa: E402
from rlinf.models.embodiment.gr00t.embodiment_tags import EmbodimentTag  # noqa: E402

B = 2
STATE_DIM = 43
ACTION_DIM = 35

# Authoritative modality.json slices (independent copy, to cross-check the converters).
STATE_SLICES = {
    "left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22),
    "left_hand": (22, 29), "right_arm": (29, 36), "right_hand": (36, 43),
}
ACTION_SLICES = {
    "left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 21), "right_hand": (21, 28),
    "waist": (28, 31), "base_height_command": (31, 32), "navigate_command": (32, 35),
}
ACTION_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand", "waist",
                "base_height_command", "navigate_command")

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILURES.append(name)


def test_obs_converter() -> None:
    # distinct value per state index so we can verify exact slice placement
    states = torch.arange(B * STATE_DIM, dtype=torch.float32).reshape(B, STATE_DIM)
    env_obs = {
        "states": states,
        "main_images": torch.randint(0, 256, (B, 8, 8, 3), dtype=torch.uint8),
        "task_descriptions": ["Lift the drill from the table."] * B,
    }
    out = sio.convert_isaaclab_arena_g1_obs_to_gr00t_format(env_obs)

    expected_keys = {f"state.{g}" for g in STATE_SLICES} | {
        "video.ego_view", "annotation.human.task_description"
    }
    check("obs.keys", set(out) == expected_keys, f"keys={sorted(out)}")
    check("obs.no_wrist_pose",
          not any("wrist_pose" in k for k in out), "eef_pose must be omitted at inference")
    check("obs.annotation_key_no_action_infix",
          "annotation.human.task_description" in out
          and "annotation.human.action.task_description" not in out)

    # each state group must be the right slice, shaped [B, 1, dim]
    ok_groups = True
    for g, (s, e) in STATE_SLICES.items():
        v = out[f"state.{g}"]
        if v.shape != (B, 1, e - s) or not np.array_equal(v[:, 0, :], states[:, s:e].numpy()):
            ok_groups = False
    check("obs.state_groups_sliced_correctly", ok_groups)

    vid = out["video.ego_view"]
    check("obs.video_shape_BT1HWC",
          vid.shape == (B, 1, 8, 8, 3) and vid.dtype == np.uint8, f"shape={vid.shape}")


def test_action_converter() -> None:
    # build group dict with distinct per-group constants, T=4 horizon
    T = 4
    chunk = {f"action.{g}": np.full((B, T, e - s), i + 1, dtype=np.float32)
             for i, (g, (s, e)) in enumerate(ACTION_SLICES.items())}
    out = sio.convert_to_isaaclab_arena_g1_action_n1d7(chunk, chunk_size=1)
    check("action.shape_35", out.shape == (B, 1, ACTION_DIM), f"shape={out.shape}")

    # verify the concat ORDER matches modality.json slices
    ok_order = True
    for i, g in enumerate(ACTION_ORDER):
        s, e = ACTION_SLICES[g]
        if not np.all(out[:, :, s:e] == i + 1):
            ok_order = False
    check("action.concat_order_matches_schema", ok_order)

    # unprefixed keys also accepted
    chunk_np = {g: np.zeros((B, T, e - s), dtype=np.float32)
                for g, (s, e) in ACTION_SLICES.items()}
    out2 = sio.convert_to_isaaclab_arena_g1_action_n1d7(chunk_np, chunk_size=1)
    check("action.unprefixed_keys_ok", out2.shape == (B, 1, ACTION_DIM))

    # missing group raises
    raised = False
    try:
        bad = dict(chunk); bad.pop("action.waist")
        sio.convert_to_isaaclab_arena_g1_action_n1d7(bad, chunk_size=1)
    except KeyError:
        raised = True
    check("action.missing_group_raises", raised)


def test_round_trip() -> None:
    # STATE: assemble (flat) -> obs converter split -> reassemble in slice order == flat
    flat_state = torch.arange(B * STATE_DIM, dtype=torch.float32).reshape(B, STATE_DIM)
    out = sio.convert_isaaclab_arena_g1_obs_to_gr00t_format(
        {"states": flat_state,
         "main_images": torch.zeros(B, 4, 4, 3, dtype=torch.uint8),
         "task_descriptions": [""] * B})
    reassembled = np.concatenate([out[f"state.{g}"][:, 0, :] for g in STATE_SLICES], axis=-1)
    check("roundtrip.state_lossless",
          np.array_equal(reassembled, flat_state.numpy()),
          "split then re-concat must equal the original 43-d state")

    # ACTION: flat 35-d -> split by schema -> action converter -> flat (identity)
    flat_action = np.arange(B * 1 * ACTION_DIM, dtype=np.float32).reshape(B, 1, ACTION_DIM)
    chunk = {f"action.{g}": flat_action[:, :, s:e] for g, (s, e) in ACTION_SLICES.items()}
    out_a = sio.convert_to_isaaclab_arena_g1_action_n1d7(chunk, chunk_size=1)
    check("roundtrip.action_lossless",
          np.array_equal(out_a, flat_action),
          "split-by-schema then converter-concat must equal the original 35-d action")


def test_registration() -> None:
    check("reg.obs_registered", sio.OBS_CONVERSION.get("isaaclab_arena_g1")
          is sio.convert_isaaclab_arena_g1_obs_to_gr00t_format)
    check("reg.action_registered", sio.ACTION_CONVERSION_N1D7.get("isaaclab_arena_g1")
          is sio.convert_to_isaaclab_arena_g1_action_n1d7)
    check("reg.embodiment_tag", EmbodimentTag.ISAACLAB_ARENA_G1.value == "isaaclab_arena_g1")

    from rlinf.envs.action_utils import prepare_actions
    act = np.random.randn(B, 1, ACTION_DIM).astype(np.float32)
    passed = prepare_actions(
        raw_chunk_actions=act.copy(), env_type="isaaclab_arena",
        model_type="gr00t_n1d7", num_action_chunks=1, action_dim=ACTION_DIM,
    )
    check("reg.prepare_actions_passthrough",
          np.array_equal(np.asarray(passed), act),
          "35-d G1 action must pass through unchanged")


def main() -> int:
    test_obs_converter()
    test_action_converter()
    test_round_trip()
    test_registration()
    print()
    if _FAILURES:
        print(f"S2.3 GR00T-ARENA-ADAPTER FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("S2.3 GR00T-ARENA-ADAPTER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
