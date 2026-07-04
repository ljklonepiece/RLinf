#!/usr/bin/env python3
"""Idempotently patch a PUBLIC Isaac-GR00T checkout for the drill_lift BC finetune.

These are the node-local source edits that must be re-applied on every fresh OSMO node
(the /workspace checkout is ephemeral; Lustre assets like the base model config.json and the
staged backbone persist and are handled by bc_finetune_drill_lift_osmo.sh, not here).

Edits (all idempotent; safe to re-run):
  gr00t/experiment/launch_finetune.py
    - backbone path        : model_name <- env GR00T_BACKBONE_PATH (default unchanged)
    - action horizon       : config.model.action_horizon <- env GR00T_ACTION_HORIZON (G1 needs 50)
    - skip stale 1st frame : config.data.skip_first_n_frames <- env GR00T_SKIP_FIRST_N_FRAMES
  gr00t/configs/data/data_config.py            : add skip_first_n_frames field (default 0)
  gr00t/data/dataset/factory.py                : thread skip_first_n_frames into the dataset
  gr00t/data/dataset/sharded_single_step_dataset.py : honor skip_first_n_frames
       (drop the first N steps of EVERY episode -- the teleop/rollout recorder logs a STALE
        first frame, an off-by-one between observation and action/reward, so the first
        transition is corrupted; gr00t's offline-RL pipeline fixes this with skip_first_n_frames=1)
  examples/finetune.sh                          : SAVE_TOTAL_LIMIT env (default 5) so we can keep
                                                  the full checkpoint sweep for selection

Usage: python3 osmo_patch_isaac_gr00t.py [/path/to/Isaac-GR00T]   (default /workspace/Isaac-GR00T)
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/Isaac-GR00T")


def edit(rel: str, old: str, new: str, token: str) -> None:
    p = ROOT / rel
    s = p.read_text()
    if token in s:
        print(f"  skip   [{token}]  (already in {rel})")
        return
    assert old in s, f"anchor not found in {rel} for [{token}]:\n{old!r}"
    p.write_text(s.replace(old, new, 1))
    print(f"  patch  [{token}]  -> {rel}")


def main() -> None:
    print(f"Patching Isaac-GR00T at {ROOT}")

    lf = "gr00t/experiment/launch_finetune.py"
    edit(lf,
         '    config.model.model_name = "nvidia/Cosmos-Reason2-2B"',
         '    config.model.model_name = os.environ.get("GR00T_BACKBONE_PATH", "nvidia/Cosmos-Reason2-2B")',
         "GR00T_BACKBONE_PATH")
    edit(lf,
         "    config.model.use_relative_action = True\n",
         '    config.model.use_relative_action = True\n'
         '    config.model.action_horizon = int(os.environ.get("GR00T_ACTION_HORIZON", config.model.action_horizon))\n',
         "GR00T_ACTION_HORIZON")
    edit(lf,
         "    config.data.ds_weights_alpha = ft_config.ds_weights_alpha\n",
         "    config.data.ds_weights_alpha = ft_config.ds_weights_alpha\n"
         '    config.data.skip_first_n_frames = int(os.environ.get("GR00T_SKIP_FIRST_N_FRAMES", config.data.skip_first_n_frames))\n',
         "GR00T_SKIP_FIRST_N_FRAMES")

    edit("gr00t/configs/data/data_config.py",
         "    num_shards_per_epoch: int = int(1e5)\n",
         "    num_shards_per_epoch: int = int(1e5)\n"
         "    skip_first_n_frames: int = 0  # drop stale first frame(s) per episode (recorder off-by-one)\n",
         "skip_first_n_frames")

    edit("gr00t/data/dataset/factory.py",
         "                    episode_sampling_rate=self.config.data.episode_sampling_rate,\n"
         "                    seed=self.config.data.seed,\n",
         "                    episode_sampling_rate=self.config.data.episode_sampling_rate,\n"
         "                    seed=self.config.data.seed,\n"
         "                    skip_first_n_frames=self.config.data.skip_first_n_frames,\n",
         "skip_first_n_frames")

    sds = "gr00t/data/dataset/sharded_single_step_dataset.py"
    edit(sds,
         "        seed: int = 42,\n        allow_padding: bool = False,\n",
         "        seed: int = 42,\n        skip_first_n_frames: int = 0,\n        allow_padding: bool = False,\n",
         "skip_first_n_frames: int = 0,")
    edit(sds,
         "        self.episode_sampling_rate = episode_sampling_rate\n",
         "        self.episode_sampling_rate = episode_sampling_rate\n"
         "        self.skip_first_n_frames = skip_first_n_frames\n",
         "self.skip_first_n_frames")
    edit(sds,
         "            step_indices = np.arange(0, self.get_effective_episode_length(ep_idx))",
         "            step_indices = np.arange(self.skip_first_n_frames, self.get_effective_episode_length(ep_idx))",
         "np.arange(self.skip_first_n_frames")

    edit("examples/finetune.sh",
         "    --save_total_limit 5\n",
         '    --save_total_limit "${SAVE_TOTAL_LIMIT:-5}"\n',
         "SAVE_TOTAL_LIMIT")

    # --- drill_lift WITH-EEF experiment: add wrist (eef) pose to the UNITREE_G1 STATE ---
    # GATED behind env GR00T_ADD_EEF=1. Default (no-eef) is the clean baseline that matches the
    # drill_lift_rl parquet (which has NO observation.eef_pose column). The eef experiment feeds
    # the hand Cartesian pose; adding it to the UNITREE_G1 modality would break a no-eef run (the
    # dataloader would try to read the missing eef column). Model max_state_dim=132 fits 43+14=57.
    if os.environ.get("GR00T_ADD_EEF", "0") == "1":
        ec = ROOT / "gr00t/configs/data/embodiment_configs.py"
        ecs = ec.read_text()
        if "left_wrist_pose" not in ecs:
            ecs += (
                "\n\n# drill_lift with-eef experiment: add wrist pose to UNITREE_G1 state modality\n"
                "_g1eef = MODALITY_CONFIGS.get(\"unitree_g1_full_body_with_waist_height_nav_cmd\")\n"
                "if _g1eef is not None and \"left_wrist_pose\" not in _g1eef[\"state\"].modality_keys:\n"
                "    _g1eef[\"state\"].modality_keys = _g1eef[\"state\"].modality_keys + [\n"
                "        \"left_wrist_pose\", \"right_wrist_pose\",\n"
                "    ]\n"
            )
            ec.write_text(ecs)
            print("  patch  [eef-state]  -> gr00t/configs/data/embodiment_configs.py")
        else:
            print("  skip   [eef-state]  (already present)")
    else:
        print("  skip   [eef-state]  (GR00T_ADD_EEF!=1; clean no-eef baseline)")

    print("Done.")


if __name__ == "__main__":
    main()
