#!/usr/bin/env python3
"""S2.2 unit check: model-agnostic DSRL in the SAC worker + a real SAC update.

Two parts, both runnable without the 3B checkpoint / GPU / Ray:

Part A - ``EmbodiedSACFSDPPolicy._resolve_use_dsrl`` (the S2.2 change) resolves the DSRL
  flag from EITHER the openpi key (``model.openpi.use_dsrl``) OR the GR00T key
  (``model.rl_head_config.use_dsrl``) OR a top-level ``model.use_dsrl`` -> so the one SAC
  worker drives both models. Tested on the REAL worker class via ``__new__``.

Part B - one full SAC update (critic + actor) on the GR00T DSRL model, exercising the
  same calls the worker makes (``sac_forward`` / ``sac_q_forward``) and the same
  optimizer param-name split (critic = critic_*+q_head, actor = the rest). Verifies:
  the param filters cleanly partition the trainable DSRL params; the frozen base is in
  neither group; losses are finite; DSRL params move while the frozen base does not.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s2_2_sac_worker_unit.py
"""

from __future__ import annotations

import sys

import torch
from omegaconf import OmegaConf

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (  # noqa: E402
    GR00T_N1_7_ForRLActionPrediction as Gr00tRL,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import (  # noqa: E402
    EmbodiedSACFSDPPolicy,
)

B = 4
ACTION_DIM = 35
ACTION_HORIZON = 50
STATE_DIM = 43
NUM_Q = 6
IMG_HW = (480, 640, 3)
GAMMA = 0.99
ALPHA = 0.1

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILURES.append(name)


# --------------------------------------------------------------------------- #
# Part A: model-agnostic use_dsrl resolution on the REAL worker class
# --------------------------------------------------------------------------- #
def _resolve(model_cfg: dict) -> bool:
    w = EmbodiedSACFSDPPolicy.__new__(EmbodiedSACFSDPPolicy)
    w.cfg = OmegaConf.create({"actor": {"model": model_cfg}})
    return w._resolve_use_dsrl()


def test_resolve_use_dsrl() -> None:
    check("resolve.gr00t_true", _resolve({"rl_head_config": {"use_dsrl": True}}) is True)
    check("resolve.openpi_true", _resolve({"openpi": {"use_dsrl": True}}) is True)
    check("resolve.toplevel_true", _resolve({"use_dsrl": True}) is True)
    check("resolve.gr00t_false", _resolve({"rl_head_config": {"use_dsrl": False}}) is False)
    check("resolve.empty_false", _resolve({}) is False)
    # openpi present but off, gr00t on -> still True (any source)
    check("resolve.mixed_true",
          _resolve({"openpi": {"use_dsrl": False}, "rl_head_config": {"use_dsrl": True}}) is True)


# --------------------------------------------------------------------------- #
# Part B: one SAC update on the GR00T DSRL model
# --------------------------------------------------------------------------- #
class _StubBase(torch.nn.Module):
    def __init__(self, **attrs):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        for k, v in attrs.items():
            setattr(self, k, v)


def _make_dsrl_model() -> Gr00tRL:
    model = Gr00tRL.__new__(Gr00tRL)
    torch.nn.Module.__init__(model)
    model.action_dim = ACTION_DIM
    model.backbone = _StubBase()
    model.action_head = _StubBase(
        model_action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON
    )
    rl_head_config = {
        "use_dsrl": True,
        "dsrl_action_noise_dim": ACTION_DIM,
        "dsrl_state_dim": STATE_DIM,
        "dsrl_num_q_heads": NUM_Q,
    }
    model.rl_head_config = rl_head_config
    model._init_dsrl_components(rl_head_config)
    model.float()
    return model


def _obs() -> dict:
    return {
        "main_images": torch.randint(0, 256, (B, *IMG_HW), dtype=torch.uint8),
        "states": torch.randn(B, STATE_DIM),
    }


# Mirror the worker's DSRL optimizer param-name filters.
_CRITIC_KEYS = ("critic_image_encoder", "critic_state_encoder", "q_head")
_ACTOR_KEYS = ("dsrl_action_noise_net", "actor_image_encoder", "actor_state_encoder")


def test_sac_update() -> None:
    torch.manual_seed(0)
    model = _make_dsrl_model()
    model.freeze_vlm()  # freeze base; only DSRL nets train (what the worker relies on)

    named = list(model.named_parameters())
    critic_params = [p for n, p in named if any(k in n for k in _CRITIC_KEYS) and p.requires_grad]
    actor_params = [p for n, p in named if any(k in n for k in _ACTOR_KEYS) and p.requires_grad]
    base_params = [p for n, p in named if n.startswith(("backbone.", "action_head."))]

    check("filters.critic_nonempty", len(critic_params) > 0, f"n={len(critic_params)}")
    check("filters.actor_nonempty", len(actor_params) > 0, f"n={len(actor_params)}")
    check("filters.disjoint",
          {id(p) for p in critic_params}.isdisjoint({id(p) for p in actor_params}))
    check("filters.base_frozen_excluded",
          all(not p.requires_grad for p in base_params)
          and not any(id(p) in {id(q) for q in critic_params + actor_params} for p in base_params))

    opt_q = torch.optim.Adam(critic_params, lr=1e-3)
    opt_pi = torch.optim.Adam(actor_params, lr=1e-3)

    obs, next_obs = _obs(), _obs()
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    rewards = torch.rand(B, 1)
    dones = torch.zeros(B, 1)

    # snapshots to prove who moves and who doesn't
    q_before = critic_params[0].detach().clone()
    pi_before = actor_params[0].detach().clone()
    base_lin = model.backbone.lin.weight.detach().clone()

    # ---- critic update (same shape as worker.forward_critic) ----
    with torch.no_grad():
        next_noise, next_logp, _ = model.sac_forward(next_obs, mode="train")
        q_next = model.sac_q_forward(next_obs, actions=next_noise)        # [B, num_q]
        q_next_agg = q_next.min(dim=1, keepdim=True).values               # clipped double-Q
        target = rewards + GAMMA * (1 - dones) * (
            q_next_agg - ALPHA * next_logp.unsqueeze(-1)
        )
    q = model.sac_q_forward(obs, actions=actions)                          # [B, num_q]
    critic_loss = torch.nn.functional.mse_loss(q, target.expand_as(q))
    opt_q.zero_grad()
    critic_loss.backward()
    opt_q.step()

    # ---- actor update (same shape as worker.forward_actor) ----
    noise, logp, _ = model.sac_forward(obs, mode="train")
    q_pi = model.sac_q_forward(obs, actions=noise, detach_encoder=True)
    q_pi_agg = q_pi.min(dim=1, keepdim=True).values
    actor_loss = (ALPHA * logp.unsqueeze(-1) - q_pi_agg).mean()
    opt_pi.zero_grad()
    actor_loss.backward()
    opt_pi.step()

    check("update.critic_loss_finite", torch.isfinite(critic_loss).item(),
          f"critic_loss={critic_loss.item():.4f}")
    check("update.actor_loss_finite", torch.isfinite(actor_loss).item(),
          f"actor_loss={actor_loss.item():.4f}")
    check("update.critic_params_moved",
          not torch.allclose(q_before, critic_params[0].detach()))
    check("update.actor_params_moved",
          not torch.allclose(pi_before, actor_params[0].detach()))
    check("update.base_unchanged",
          torch.allclose(base_lin, model.backbone.lin.weight.detach()),
          "frozen base must not move")


def main() -> int:
    test_resolve_use_dsrl()
    test_sac_update()
    print()
    if _FAILURES:
        print(f"S2.2 SAC-WORKER FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("S2.2 SAC-WORKER OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
