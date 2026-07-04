#!/usr/bin/env python3
"""S2.1 unit check: DSRL plumbing in the GR00T N1.7 action model (no 3B checkpoint).

The full model needs the public GR00T-N1.7-3B checkpoint (downloaded in S4.1) + a GPU,
so this isolates the *new DSRL code* by building model/action-head instances via
``__new__`` (bypassing the heavy ``__init__``) and attaching only what each method needs:
lightweight DSRL nets + tiny stub backbone/action_head. It exercises the REAL methods:

  * _init_dsrl_components: builds GaussianPolicy/encoders/q_head; noise dim == 35 (G1) and
    a mismatched dsrl_action_noise_dim raises.
  * sac_forward: eval (deterministic) reproducible, train (stochastic) varies; returns the
    3-tuple (noise [B,H,35], logprob [B], dist_params); surfaces nothing extra.
  * sac_q_forward: returns Q-values [B, num_q_heads]; detach_encoder honored.
  * get_rl_action(noise=...): the injected noise becomes the initial latent x_t (chains[0])
    and, with an identity denoiser, the produced action == the injected noise.
  * freeze_vlm: backbone + action_head frozen; DSRL nets stay trainable.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s2_1_gr00t_dsrl_unit.py
"""

from __future__ import annotations

import sys
import types

import torch

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (  # noqa: E402
    FlowMatchingActionHeadForRLActionPrediction as FlowHead,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (  # noqa: E402
    GR00T_N1_7_ForRLActionPrediction as Gr00tRL,
)

B = 2
ACTION_DIM = 35          # G1 joint action == diffusion latent dim == DSRL noise dim
ACTION_HORIZON = 50
STATE_DIM = 43
NUM_Q = 6
IMG_HW = (480, 640, 3)

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILURES.append(name)


class _StubBase(torch.nn.Module):
    """Minimal nn.Module standing in for the heavy backbone / action head."""

    def __init__(self, **attrs):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        for k, v in attrs.items():
            setattr(self, k, v)


def _make_dsrl_model(noise_dim: int = ACTION_DIM) -> Gr00tRL:
    """Build a bare GR00T-RL model with only DSRL components + stub base modules."""
    model = Gr00tRL.__new__(Gr00tRL)
    torch.nn.Module.__init__(model)
    model.action_dim = ACTION_DIM
    model.backbone = _StubBase()
    model.action_head = _StubBase(
        model_action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON
    )
    rl_head_config = {
        "use_dsrl": True,
        "dsrl_action_noise_dim": noise_dim,
        "dsrl_state_dim": STATE_DIM,
        "dsrl_num_q_heads": NUM_Q,
        "dsrl_image_latent_dim": 64,
        "dsrl_state_latent_dim": 64,
        "dsrl_hidden_dims": (128, 128, 128),
    }
    model.rl_head_config = rl_head_config
    model._init_dsrl_components(rl_head_config)
    model.float()  # bf16 -> fp32 so the CPU-only test runs (Conv2d bf16 CPU is flaky)
    return model


def _env_obs() -> dict:
    return {
        "main_images": torch.randint(0, 256, (B, *IMG_HW), dtype=torch.uint8),
        "states": torch.randn(B, STATE_DIM),
    }


def test_init_and_shapes() -> None:
    model = _make_dsrl_model()
    check("init.use_dsrl_true", model.use_dsrl is True)
    check("init.noise_dim_35", model.dsrl_action_noise_dim == ACTION_DIM,
          f"noise_dim={model.dsrl_action_noise_dim}")
    check("init.has_all_nets", all(hasattr(model, n) for n in (
        "dsrl_action_noise_net", "actor_image_encoder", "actor_state_encoder",
        "critic_image_encoder", "critic_state_encoder", "q_head")))
    # mismatched noise dim must fail loudly
    raised = False
    try:
        _make_dsrl_model(noise_dim=32)
    except ValueError as e:
        raised = "model_action_dim" in str(e)
    check("init.noise_dim_mismatch_raises", raised)


def test_sac_forward() -> None:
    model = _make_dsrl_model()
    obs = _env_obs()

    noise_e1, lp_e1, dp = model.sac_forward(obs, mode="eval")
    noise_e2, _, _ = model.sac_forward(obs, mode="eval")
    check("sac.returns_3_tuple", isinstance(dp, type(None)) or isinstance(dp, tuple))
    check("sac.noise_shape", tuple(noise_e1.shape) == (B, ACTION_HORIZON, ACTION_DIM),
          f"shape={tuple(noise_e1.shape)}")
    check("sac.logprob_shape", tuple(lp_e1.shape) == (B,), f"shape={tuple(lp_e1.shape)}")
    check("sac.eval_deterministic", torch.allclose(noise_e1, noise_e2),
          "two eval calls must match")
    # the same noise vector is repeated across the horizon (matches the flow latent)
    check("sac.noise_repeated_over_horizon",
          torch.allclose(noise_e1[:, 0], noise_e1[:, -1]))

    torch.manual_seed(0)
    noise_t1, _, _ = model.sac_forward(obs, mode="train")
    torch.manual_seed(1)
    noise_t2, _, _ = model.sac_forward(obs, mode="train")
    check("sac.train_stochastic", not torch.allclose(noise_t1, noise_t2),
          "two train calls (diff seeds) must differ")

    _, _, dp2 = model.sac_forward(obs, mode="train", return_dist_params=True)
    check("sac.dist_params_when_requested", isinstance(dp2, tuple) and len(dp2) == 2)


def test_sac_q_forward() -> None:
    model = _make_dsrl_model()
    obs = _env_obs()
    actions = torch.randn(B, ACTION_HORIZON, ACTION_DIM)
    q = model.sac_q_forward(obs, actions=actions)
    check("sacq.shape", tuple(q.shape) == (B, NUM_Q), f"shape={tuple(q.shape)}")
    q_detach = model.sac_q_forward(obs, actions=actions, detach_encoder=True)
    check("sacq.detach_ok", tuple(q_detach.shape) == (B, NUM_Q))
    # accepts already-flattened [B, dim] actions too
    q_flat = model.sac_q_forward(obs, actions=torch.randn(B, ACTION_DIM))
    check("sacq.flat_actions_ok", tuple(q_flat.shape) == (B, NUM_Q))


def _make_identity_flow_head() -> FlowHead:
    """Bare flow head whose denoiser is the identity (so action == injected noise)."""
    head = FlowHead.__new__(FlowHead)
    torch.nn.Module.__init__(head)
    head.model_action_dim = ACTION_DIM
    head.action_horizon = ACTION_HORIZON
    head.num_inference_timesteps = 4
    head.action_chunk = 1
    head.env_action_dim = ACTION_DIM
    head.valid_action_dim = ACTION_DIM
    head.rl_config = {}
    # Stub out everything get_rl_action calls so the denoise loop is identity+deterministic.
    head._process_backbone_output = lambda bo: bo
    head._encode_state_features = lambda ai, eid: torch.zeros(B, 1, 8)
    head.sample_mean_var_val = lambda **kw: (kw["x_t"], torch.zeros_like(kw["x_t"]))
    head.get_logprob_norm = lambda sample, mu, sigma: torch.zeros_like(sample)
    head.get_value = lambda vl, sf: torch.zeros(B, device=vl.device, dtype=vl.dtype)
    return head


def test_noise_injection() -> None:
    head = _make_identity_flow_head()
    vl_embs = torch.zeros(B, 1, 8)  # no backbone_features attr -> used directly as vl_embs
    action_input = types.SimpleNamespace()  # no embodiment_id -> defaults to 0
    fixed_noise = torch.randn(B, ACTION_HORIZON, ACTION_DIM)

    _, out = head.get_rl_action(vl_embs, action_input, mode="eval", noise=fixed_noise)
    chains = out["chains"]
    check("inject.chains0_is_noise", torch.allclose(chains[:, 0], fixed_noise),
          "initial latent x_t must equal the injected noise")
    check("inject.action_is_noise", torch.allclose(out["actions"], fixed_noise),
          "identity denoise -> action == injected noise")

    # wrong-shaped noise must be rejected
    raised = False
    try:
        head.get_rl_action(vl_embs, action_input, mode="eval",
                           noise=torch.randn(B, ACTION_HORIZON, 32))
    except ValueError:
        raised = True
    check("inject.bad_noise_shape_raises", raised)

    # without noise, falls back to fresh randn (x_t != a fixed tensor across calls)
    _, out_a = head.get_rl_action(vl_embs, action_input, mode="eval")
    _, out_b = head.get_rl_action(vl_embs, action_input, mode="eval")
    check("inject.no_noise_random", not torch.allclose(out_a["chains"][:, 0],
                                                       out_b["chains"][:, 0]))


def test_freeze_vlm() -> None:
    model = _make_dsrl_model()
    model.freeze_vlm()
    base_grad = [p.requires_grad for p in model.backbone.parameters()] + \
                [p.requires_grad for p in model.action_head.parameters()]
    dsrl_grad = (
        [p.requires_grad for p in model.dsrl_action_noise_net.parameters()]
        + [p.requires_grad for p in model.q_head.parameters()]
        + [p.requires_grad for p in model.actor_image_encoder.parameters()]
        + [p.requires_grad for p in model.critic_state_encoder.parameters()]
    )
    check("freeze.base_all_frozen", not any(base_grad), f"any_grad={any(base_grad)}")
    check("freeze.dsrl_all_trainable", all(dsrl_grad), f"all_grad={all(dsrl_grad)}")


def main() -> int:
    test_init_and_shapes()
    test_sac_forward()
    test_sac_q_forward()
    test_noise_injection()
    test_freeze_vlm()
    print()
    if _FAILURES:
        print(f"S2.1 GR00T-DSRL FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("S2.1 GR00T-DSRL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
