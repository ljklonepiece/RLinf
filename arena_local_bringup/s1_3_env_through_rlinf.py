#!/usr/bin/env python3
"""S1.3 verify: build the Arena env THROUGH RLinf and drive it against the real server.

This is the integration check for the S1 env layer. Unlike S1.1 (which unit-tested the
transport against a fake server), this goes through RLinf's real entry path:
``get_env_cls("isaaclab_arena", env_cfg)`` -> ``IsaaclabArenaG1Env`` -> (proxy)
``ArenaSocketEnv`` -> the **real** py3.12 Arena server (LMDrillLiftRlD1, g1 embodiment)
running in the gear-n2-eval container. It asserts the canonical RLinf contract:

  * env class resolves via the RLinf registry from the env YAML
  * reset() returns torch obs with keys {main_images, states, task_descriptions} and the
    expected shapes/dtypes on the proxy device (CPU)
  * the action interface is 35-d (g1 joint-space)
  * 10 steps run; reward is a torch tensor; the episode/success term is readable
    (infos["episode"]["success_once"])

No GR00T, no DSRL -- env layer only.

Prereq: the Arena server must be up (started by hand or via the docker helper):
    ARENA_PORT=5557 ARENA_ENV=LMDrillLiftRlD1 ARENA_NUM_ENVS=1 \
      /home/juekunl/Work/arena_local_bringup/run_arena_server.sh
Then run with the RLinf py3.11 venv:
    ARENA_PORT=5557 /home/juekunl/Work/RLinf/.venv/bin/python -u \
      /home/juekunl/Work/arena_local_bringup/s1_3_env_through_rlinf.py
"""

from __future__ import annotations

import os
import sys

import torch
from omegaconf import OmegaConf, open_dict

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.envs import get_env_cls  # noqa: E402

ENV_YAML = os.path.join(
    RLINF, "examples/embodiment/config/env/isaaclab_arena_drill_lift.yaml"
)
PORT = int(os.environ.get("ARENA_PORT", "5557"))
HOST = os.environ.get("ARENA_HOST", "127.0.0.1")
STEPS = int(os.environ.get("S1_3_STEPS", "10"))

EXPECT_STATE_DIM = 43
EXPECT_ACTION_DIM = 35
EXPECT_IMG_HW = (480, 640, 3)


def _build_cfg():
    env_yaml = OmegaConf.load(ENV_YAML)
    # Root so the YAML's ${env.train.*} / ${runner.logger.log_path} interpolations resolve.
    root = OmegaConf.create(
        {"runner": {"logger": {"log_path": "/tmp/s1_3"}}, "env": {"train": env_yaml}}
    )
    cfg = root.env.train
    with open_dict(cfg):
        cfg.init_params.server.host = HOST
        cfg.init_params.server.port = PORT
    return cfg


def main() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
        if not cond:
            failures.append(name)

    cfg = _build_cfg()
    print(f"connecting to Arena server {HOST}:{PORT} ...", flush=True)

    cls = get_env_cls(cfg.env_type, env_cfg=cfg)
    check("registry.resolves_proxy", cls.__name__ == "IsaaclabArenaG1Env", cls.__name__)

    env = cls(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    check("env.action_dim_35", env.env.action_dim == EXPECT_ACTION_DIM,
          f"action_dim={env.env.action_dim}")
    check("env.device_cpu", env.device == torch.device("cpu"), f"device={env.device}")

    obs, info = env.reset()
    check("reset.obs_keys", set(obs) == {"main_images", "states", "task_descriptions"},
          f"keys={sorted(obs)}")
    mi = obs["main_images"]
    check("reset.images_torch_uint8_cpu",
          isinstance(mi, torch.Tensor) and mi.dtype == torch.uint8
          and mi.device.type == "cpu" and tuple(mi.shape) == (1, *EXPECT_IMG_HW),
          f"{type(mi).__name__} {mi.dtype} {tuple(mi.shape)} dev={mi.device}")
    st = obs["states"]
    check("reset.states_torch_43",
          isinstance(st, torch.Tensor) and tuple(st.shape) == (1, EXPECT_STATE_DIM),
          f"{type(st).__name__} {tuple(st.shape)}")
    check("reset.task_descriptions",
          isinstance(obs["task_descriptions"], list) and len(obs["task_descriptions"]) == 1,
          f"{obs['task_descriptions']}")

    action = torch.zeros((1, env.env.action_dim), dtype=torch.float32)
    rewards = []
    last_infos = {}
    ok_steps = True
    for i in range(STEPS):
        obs, reward, terminated, truncated, infos = env.step(action)
        last_infos = infos
        rewards.append(float(reward[0]))
        if not (isinstance(reward, torch.Tensor) and tuple(reward.shape) == (1,)):
            ok_steps = False
        if set(obs) != {"main_images", "states", "task_descriptions"}:
            ok_steps = False
    check("step.x{}_consistent".format(STEPS), ok_steps,
          f"rewards[:3]={[round(r, 6) for r in rewards[:3]]}")
    check("step.reward_finite", all(r == r for r in rewards), f"n={len(rewards)}")
    has_success = (
        "episode" in last_infos and "success_once" in last_infos["episode"]
    )
    check("step.success_term_readable", has_success,
          f"episode keys={list(last_infos.get('episode', {}))}")

    # Sparse-reward fix (S1.4): the env config defaults reward_mode=sparse, so reward is
    # the true 'success' term (not Arena's always-positive shaped reward). Idle (zero)
    # actions never lift the drill, so every reward must be exactly 0.0 -- and therefore
    # RLinf's inherited success metric (success_once = step_reward>0) must stay False.
    # This is the direct opposite of the old shaped behaviour (S1.3 caveat), proving the
    # fix end to end.
    check("sparse.idle_reward_all_zero", all(r == 0.0 for r in rewards),
          f"rewards={[round(r, 6) for r in rewards]}")
    if has_success:
        succ = last_infos["episode"]["success_once"]
        check("sparse.success_once_false_on_idle", not bool(succ.any()),
              f"success_once={succ.tolist()}")
        print(f"       success_once={succ.tolist()} return={last_infos['episode']['return'].tolist()}",
              flush=True)
    # NOTE: IsaaclabBaseEnv.step() rebuilds infos via _record_metrics(..., {}) and drops
    # the underlying env infos, so the client's infos['success'] does NOT reach this layer
    # by design -- RLinf's canonical success signal here is episode.success_once (now
    # correct thanks to sparse reward). The per-step infos['success'] surfacing is verified
    # at the client layer in s1_4_reward_mode_unit.py instead.

    env.close()
    check("close.no_exception", True)

    print(flush=True)
    if failures:
        print(f"S1.3 ENV-THROUGH-RLINF FAILED ({len(failures)}): {failures}", flush=True)
        return 1
    print("S1.3 ENV-THROUGH-RLINF OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
