#!/usr/bin/env python3
"""S2.4/S3 config preflight: compose + validate the two DSRL-on-GR00T configs (no ckpt).

Verifies the scaffolded configs are well-formed and the DSRL pipeline wiring resolves,
WITHOUT launching Ray / loading the 3B model / starting Arena:

  * Hydra composes the config (defaults: env + model/gr00t_n1d7 + fsdp + syncer).
  * rlinf.config.validate_cfg accepts it.
  * loss_type/adv_type == embodied_sac; runner.dry_run present.
  * EmbodiedSACFSDPPolicy._resolve_use_dsrl(cfg) -> True (S2.2 closes the loop).
  * drill_lift: obs_converter_type registered in OBS_CONVERSION/ACTION_CONVERSION_N1D7
    and embodiment_tag valid (S2.3 closes the loop); action_dim 35, dsrl_state_dim 43.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s2_4_s3_config_preflight.py
"""

from __future__ import annotations

import os
import sys

RLINF = "/home/juekunl/Work/RLinf"
CONFIG_DIR = os.path.join(RLINF, "examples/embodiment/config")
os.environ.setdefault("EMBODIED_PATH", os.path.join(RLINF, "examples/embodiment"))
sys.path.insert(0, RLINF)

from hydra import compose, initialize_config_dir  # noqa: E402

from rlinf.config import validate_cfg  # noqa: E402
from rlinf.models.embodiment.gr00t import simulation_io as sio  # noqa: E402
from rlinf.models.embodiment.gr00t.embodiment_tags import EmbodimentTag  # noqa: E402
from rlinf.workers.actor.fsdp_sac_policy_worker import (  # noqa: E402
    EmbodiedSACFSDPPolicy,
)

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILURES.append(name)


def _compose(config_name: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        return compose(config_name=config_name)


def _resolve_use_dsrl(cfg) -> bool:
    w = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    w.cfg = cfg
    return w._resolve_use_dsrl()


def _check_common(tag: str, cfg) -> None:
    cfg = validate_cfg(cfg)
    check(f"{tag}.validate_cfg_ok", True)
    check(f"{tag}.loss_type_sac", cfg.algorithm.loss_type == "embodied_sac",
          f"loss_type={cfg.algorithm.loss_type}")
    check(f"{tag}.adv_type_sac", cfg.algorithm.adv_type == "embodied_sac")
    check(f"{tag}.dry_run_declared", "dry_run" in cfg.runner,
          f"runner keys include dry_run: {'dry_run' in cfg.runner}")
    check(f"{tag}.dry_run_default_false", bool(cfg.runner.dry_run) is False)
    check(f"{tag}.use_dsrl_resolves_true", _resolve_use_dsrl(cfg) is True)
    check(f"{tag}.model_is_gr00t", cfg.actor.model.model_type == "gr00t_n1d7")
    check(f"{tag}.add_value_head_false", bool(cfg.actor.model.add_value_head) is False)
    check(f"{tag}.rl_head_use_dsrl",
          bool(cfg.actor.model.rl_head_config.use_dsrl) is True)
    return cfg


def test_libero_dsrl_gr00t() -> None:
    cfg = _compose("libero_spatial_dsrl_gr00t_n1d7")
    cfg = _check_common("libero", cfg)
    check("libero.state_dim_8", int(cfg.actor.model.rl_head_config.dsrl_state_dim) == 8)
    check("libero.embodiment_libero", cfg.actor.model.embodiment_tag == "libero_sim")
    # noise dim omitted -> auto-derive
    check("libero.noise_dim_omitted",
          "dsrl_action_noise_dim" not in cfg.actor.model.rl_head_config)


def test_drill_lift_dsrl_gr00t() -> None:
    cfg = _compose("drill_lift_dsrl_gr00t_n1d7")
    cfg = _check_common("drill", cfg)
    check("drill.env_type_arena", cfg.env.train.env_type == "isaaclab_arena")
    check("drill.action_dim_35", int(cfg.actor.model.action_dim) == 35)
    check("drill.state_dim_43", int(cfg.actor.model.rl_head_config.dsrl_state_dim) == 43)
    check("drill.single_env", int(cfg.env.train.total_num_envs) == 1)

    oct_ = cfg.actor.model.obs_converter_type
    check("drill.obs_converter_value", oct_ == "isaaclab_arena_g1", f"obs_converter={oct_}")
    check("drill.obs_converter_registered", oct_ in sio.OBS_CONVERSION)
    check("drill.action_converter_registered", oct_ in sio.ACTION_CONVERSION_N1D7)
    tag = cfg.actor.model.embodiment_tag
    check("drill.embodiment_tag_valid",
          tag == EmbodimentTag.ISAACLAB_ARENA_G1.value)
    # the env worker connects to the Arena socket server
    check("drill.env_server_block",
          "server" in cfg.env.train.init_params
          and int(cfg.env.train.init_params.server.port) == 5557)


def main() -> int:
    test_libero_dsrl_gr00t()
    test_drill_lift_dsrl_gr00t()
    print()
    if _FAILURES:
        print(f"S2.4/S3 CONFIG PREFLIGHT FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("S2.4/S3 CONFIG PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
