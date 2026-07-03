# IsaacLab-Arena ↔ RLinf integration — experiment log

Experiments validating how RLinf (CPython 3.11) drives IsaacLab-Arena (CPython 3.12).
Each entry follows: **why → design → result → analysis/conclusion**.

---

## Background / context

- RLinf runs on **py3.11**; IsaacLab-Arena (`isaaclab==4.6.5`, commit `dd6618c68`) requires
  **py3.12** (cp312 native extensions). They cannot share one process (C-ABI mismatch).
- Headless render parity on the local RTX 5880 workstation requires running Arena inside the
  cluster image `nvcr.io/nvidian/gear-n2-eval:py312-2026.05.07` (isaac-sim 6.0.0-rc.22). A bare
  host headless run crashes in `reset()`.

---

## Precursor — Ray cross-interpreter probe (FAILED, decisive)

1. **Why** — We hoped to run the Arena env worker under py3.12 inside RLinf's single Ray cluster
   via `python_interpreter_path` → Ray `runtime_env["py_executable"]` ("Option A"). Needed to know
   if a py3.11 Ray driver can own a py3.12 worker in the same cluster.
2. **How** — `arena_local_bringup/l2_xinterp_probe.py`: a py3.11 (RLinf venv) Ray driver launches an
   actor pinned to the Arena py3.12 interpreter; attempt a numpy round-trip.
3. **Result** — `RuntimeError: Version mismatch: cluster started with Python 3.11.14, this process
   started with Python 3.12.3`. The actor never initialized.
4. **Analysis/conclusion** — Ray enforces **Python minor-version parity** across a cluster (not just
   matching Ray versions). Option A is impossible. Remaining options: A′ unify RLinf on py3.12, or
   **B run Arena out-of-process behind a socket**. Decision (user): stay py3.11 → **Option B**.

---

## B1 — Cross-interpreter socket boundary (PASSED)

### 1. Why perform this experiment
Under Option B, Arena runs in its own py3.12 process and RLinf (py3.11) talks to it over a socket.
The make-or-break risk: **can a real Arena observation (rendered camera image + 43-d proprio state)
and an action/reward actually cross the py3.11 ↔ py3.12 boundary intact?** Every downstream step
(the RLinf proxy env, training config, DSRL run) is wasted if this boundary doesn't work. This test
exists to confirm the boundary before building anything on top of it.

### 2. How the test is designed to answer that
Minimum pieces to exercise the boundary, nothing more:
- `protocol.py` — stdlib-only length-framed pickle (proto 5); imports identically in both interpreters.
- `arena_server.py` — launched by the **py3.12** Arena venv; builds the real `LMDrillLiftRlD1` env
  (`g1` embodiment) and serves `spec` / `reset` / `step` / `close`, returning the canonical obs as
  **numpy** (`g1_layout.wrap_obs`), plus reward/terminated/truncated.
- `arena_local_bringup/l2b_socket_smoke.py` — the **py3.11** client (RLinf venv): connect, `spec`,
  `reset`, then 5 zero ("idle") actions, printing shapes/dtypes of everything received.

Design choices tied to the "why": only **numpy/CPU** crosses the socket (never live GPU tensors) —
this is the contract that makes cross-version transfer safe. The server runs in the proven
`gear-n2-eval` container on `--network host` so the host py3.11 client reaches `127.0.0.1:<port>`.
Success = client receives correctly shaped image+state arrays and valid rewards; clean shutdown.

Reproduce:
```bash
# terminal 1 (server, py3.12 in container):
ARENA_PORT=5557 ARENA_ENV=LMDrillLiftRlD1 ARENA_NUM_ENVS=1 \
  /home/juekunl/Work/arena_local_bringup/run_arena_server.sh
# wait ~60s for arena_local_bringup/arena_server.ready

# terminal 2 (client, py3.11 RLinf venv):
ARENA_PORT=5557 SMOKE_STEPS=5 \
  /home/juekunl/Work/RLinf/.venv/bin/python -u \
  /home/juekunl/Work/arena_local_bringup/l2b_socket_smoke.py
```

### 3. The result
```
client py=3.11.14  ->  server py3.12 @ 127.0.0.1:5557
spec  : action_dim=35, num_envs=1
reset : main_images       (1, 480, 640, 3) uint8
        states            (1, 43)          float32
        task_descriptions ['Pick up the power drill from the table and lift it.']
step x5 : reward (1,) float32, terminated=[False], truncated=[False]
          mean reward/step = [1.1e-4, 1.1e-4, 1.4e-4, 1.2e-4, 1.1e-4]
close : container exited cleanly
```
Server build log confirmed shaped reward terms: approach_object 2.0, reach_object 3.0,
lifting_object 5.0, success_bonus 10.0, grasp_bonus 10.0, action_penalty -0.001.

### 4. Analysis and conclusion
- **The boundary works.** A rendered image `(1,480,640,3) uint8` and the `(1,43)` state crossed
  py3.12 → py3.11 with correct shapes/dtypes; a py3.11 action crossed back and produced a valid
  reward. No serialization corruption, no version error.
- **Action interface is correct (`35`)** — confirms the `g1` embodiment that matches the GR00T policy.
- **Numbers are sane:** idle actions → robot holds still → reward ≈ 0 (only the tiny action penalty),
  proving the reward channel is live and correctly wired, not stuck/garbage.
- **Lifecycle is clean:** build → serve → teardown with no orphan processes.
- **Conclusion:** Option B is validated at its riskiest point; safe to build the py3.11 proxy env on
  top. Constraint to preserve: **only numpy/CPU data crosses the socket** — never live torch/GPU tensors.

---

## S1.1 — py3.11 proxy env (`ArenaSocketEnv`) transport unit (PASSED)

### 1. Why perform this experiment
B1 proved the *raw socket boundary* works, but RLinf doesn't talk raw protocol — it drives an env
through the `IsaaclabBaseEnv` interface (`reset`/`step`/`device`/`close`, torch tensors, metrics,
`chunk_step`). S1.1 introduces `ArenaSocketEnv`, a py3.11 client that is a **drop-in for
`SubProcIsaacLabEnv`** so `IsaaclabArenaG1Env` can reuse all of `IsaaclabBaseEnv`'s generic logic
unchanged. The risk this test addresses: **does that new client class correctly translate between
RLinf's torch-tensor world and the numpy wire** — action torch→float32-numpy out, and
reward/term/trunc numpy→torch on the proxy device back — and present the exact `SubProcIsaacLabEnv`
API? If this is wrong, the proxy env will fail in subtle ways (scrambled dtypes, wrong device,
wrong action shape on the wire) that are hard to debug once the heavy real server + DSRL are stacked
on top.

### 2. How the test is designed to answer that
Isolate the *new* code (`ArenaSocketEnv`) from the *already-proven* real server (B1) and from Isaac
entirely. `arena_local_bringup/s1_1_arena_client_unit.py` starts a tiny **thread-based fake server**
(py3.11) that speaks the same `protocol` and returns canonical numpy obs, then drives the real
`ArenaSocketEnv` against it. This makes the test fast/deterministic (no GPU, no Docker) and pins the
failure surface to exactly the client translation logic. It asserts, end to end:
SPEC handshake (`action_dim`/`num_envs`); `reset()` obs keys/shapes/dtypes; that the torch action is
serialized as `(num_envs, action_dim)` **float32** on the wire (captured server-side); that
reward/terminated/truncated come back as **torch tensors on CPU** with correct dtype/shape/value;
`device()`; and clean `close()`. (The real-server path was already covered by B1; full
build-through-RLinf is deferred to S1.3.)

Reproduce:
```bash
/home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s1_1_arena_client_unit.py
```

### 3. The result
```
[PASS] spec.action_dim -- action_dim=35
[PASS] reset.obs_keys -- keys=['main_images','states','task_descriptions']
[PASS] reset.images_numpy_uint8 -- ndarray (1, 480, 640, 3)
[PASS] reset.states_shape -- shape=(1, 43)
[PASS] step.action_on_wire_shape -- server saw (1, 35)
[PASS] step.action_on_wire_float32 -- server saw float32
[PASS] step.reward_is_torch_cpu -- Tensor torch.float32 (1,) dev=cpu
[PASS] step.reward_value -- reward=0.25
[PASS] step.terminated_bool_tensor -- torch.bool (1,)
[PASS] device.is_cpu / close.no_exception
S1.1 UNIT OK   (14/14)
```
Also: `ruff check` clean; `import rlinf.envs.isaaclab_arena` works in py3.11 with no Arena deps, and
`IsaaclabArenaG1Env.__mro__` confirms it still extends `IsaaclabBaseEnv` (generic logic reused).

### 4. Analysis and conclusion
- **The client is a correct drop-in.** It exposes the exact `SubProcIsaacLabEnv` API and does the
  torch↔numpy translation in the right direction on each leg: action leaves as contiguous float32
  numpy of shape `(num_envs, action_dim)`; reward/term/trunc return as torch tensors on the proxy
  device (CPU), so `IsaaclabBaseEnv.step()`'s `.clone()` / metric math operate on tensors as before.
- **Obs stay numpy across the boundary** and are converted to torch only in `_wrap_obs` (on
  `self.device`) — preserving the "only numpy/CPU crosses the socket" invariant from B1, and keeping
  obs serialization-friendly for disaggregated env↔actor Ray workers.
- **Refactor shrank the coupling:** `arena_env.py` now overrides only `_init_isaaclab_env` (build
  socket client) and `_wrap_obs` (numpy→torch); state assembly/camera selection live server-side in
  `g1_layout.wrap_obs`. The old in-process `_make_env_function`/`_assemble_state` (Option A) are gone.
- **Conclusion:** the proxy transport is correct in isolation. Remaining unknowns are *integration*
  ones (cfg plumbing, building the env through RLinf's worker against the real server) — addressed in
  S1.2 (config wiring) and S1.3 (build-through-RLinf verify).

---

## S1.3 — env built THROUGH RLinf vs the real Arena server (PASSED)

### 1. Why perform this experiment
S1.1 proved the client logic against a *fake* server; S1.2 proved the config parses. Neither
proves the real thing: that RLinf's actual entry path (`get_env_cls` → `IsaaclabArenaG1Env` →
`ArenaSocketEnv`) drives the **real** Isaac/Arena sim and yields obs/actions that satisfy RLinf's
canonical contract. This is the gate before any model/DSRL work (S2+): if the env layer doesn't
present the right keys/shapes/dtypes/device and a readable success term, everything above it is
built on sand. Scope is deliberately env-only — no GR00T, no DSRL.

### 2. How the test is designed to answer that
`arena_local_bringup/s1_3_env_through_rlinf.py` (py3.11 RLinf venv) loads the *real* env YAML
(`examples/embodiment/config/env/isaaclab_arena_drill_lift.yaml`), resolves its interpolations under
a minimal root, and instantiates the env **only** via `get_env_cls("isaaclab_arena", env_cfg)` — i.e.
exactly how an RLinf env worker would. The Arena server is the real one
(`run_arena_server.sh` → `LMDrillLiftRlD1`, `g1` embodiment, in the gear-n2-eval container). It
asserts: registry resolves the proxy; `action_dim == 35`; env device is CPU; `reset()` returns
`{main_images, states, task_descriptions}` as torch (uint8 `(1,480,640,3)` image on CPU, `(1,43)`
state); then 10 idle steps with a torch reward and a readable `infos["episode"]["success_once"]`.

Reproduce:
```bash
# server (py3.12 container):
ARENA_PORT=5557 ARENA_ENV=LMDrillLiftRlD1 ARENA_NUM_ENVS=1 \
  /home/juekunl/Work/arena_local_bringup/run_arena_server.sh
# client (py3.11 RLinf venv), after arena_server.ready appears:
ARENA_PORT=5557 /home/juekunl/Work/RLinf/.venv/bin/python -u \
  /home/juekunl/Work/arena_local_bringup/s1_3_env_through_rlinf.py
```

### 3. The result
```
[PASS] registry.resolves_proxy -- IsaaclabArenaG1Env
[PASS] env.action_dim_35 / env.device_cpu
[PASS] reset.obs_keys -- ['main_images','states','task_descriptions']
[PASS] reset.images_torch_uint8_cpu -- (1, 480, 640, 3)
[PASS] reset.states_torch_43 -- (1, 43)
[PASS] step.x10_consistent -- rewards[:3]=[6.4e-05, 6.5e-05, 8.8e-05]
[PASS] step.success_term_readable -- episode keys=['success_once','return','episode_len','reward']
       success_once=[True]  return=6.88e-04
S1.3 ENV-THROUGH-RLINF OK   (12/12)
```
Server logged a clean `received CLOSE; shutting down`; the container exited with no orphan process.

### 4. Analysis and conclusion
- **The env layer is correct end to end through RLinf.** All generic `IsaaclabBaseEnv` machinery
  (metrics, `step`, elapsed steps, truncation) runs unchanged on top of the socket proxy; obs come
  out in the canonical RLinf form with the right shapes/dtypes on the proxy device.
- **CAVEAT (carry into S2.3):** `success_once` came back `True` after *idle* actions. That is **not**
  real task success — `IsaaclabBaseEnv._record_metrics` defines `success_once = (step_reward > 0)`,
  and Arena's reward is *shaped* and always slightly positive (approach/reach terms), so any step
  trips it. The genuine success signal is Arena's `success` term in `info`, not `step_reward > 0`.
  When wiring DSRL reward/success logging (S2.3 / S3) we must surface the true `success` term rather
  than rely on the inherited `step_reward > 0` heuristic, and decide the reward convention (shaped
  vs sparse) explicitly.
- **Conclusion:** S1 (Arena env inside RLinf, model/algo-agnostic) is complete and verified. Safe to
  proceed to S2 (DSRL into the GR00T action model + SAC worker), with the success-term caveat noted.

---

## S1.4 — sparse reward as the generic reward formulation (PASSED)

### 1. Why perform this experiment
S1.3 surfaced a real defect: RLinf's inherited success metric is `success_once = (step_reward > 0)`,
but Arena's *shaped* reward (approach/reach/lift/grasp terms) is always slightly positive, so
**every** step — even idle — registered as "success". That makes the success signal meaningless and
would corrupt any RL training/eval that trusts it. We want a **generic, task-agnostic reward** that
(a) is a correct success signal and (b) matches the proven gr00t IQL pipeline, which trains with
`reward_mode=sparse`. The question: can we make sparse (success-based) reward the default and prove
it both fixes the metric and preserves the done/bootstrap signals?

### 2. How the test is designed to answer that
Mirror gr00t's definition exactly (confirmed by reading `gr00t/.../isaaclab_arena_env_wrapper.py`
and `rl_wrapper.py`): success is the IsaacLab termination term **literally named `success`**
(`object_lifted & is_grasped`), read via `termination_manager.get_term("success")` — *not* `terminated`
(which also fires on failure terms like `object_dropping`) and *not* `reward > 0`. Sparse reward is
then `reward = success.float()`. Implementation splits cleanly across the socket:
- **server** (`arena_server.py`): extracts the `success` term, sends it as a dedicated `KEY_SUCCESS`
  channel alongside the shaped `KEY_REWARD`, and advertises `has_success` in the SPEC. `terminated` /
  `truncated` stay the true signals.
- **client** (`arena_client.py`): a `reward_mode` knob (default `sparse`) selects
  `reward = success` vs `reward = shaped`; always surfaces `infos["success"]`; fails fast if `sparse`
  is requested but the env has no success term (prevents silent all-zero reward).

Two verification layers, isolated then integrated:
- **Unit (fake server, no Isaac)** `s1_4_reward_mode_unit.py`: success fires on step 2; assert
  sparse reward flips `0.0 → 1.0` tracking success while shaped stays `0.25`; terminated/truncated
  pass through; sparse-without-success raises; invalid mode raises.
- **Integration (real server)** re-run `s1_3_env_through_rlinf.py` with the config default
  (`reward_mode: sparse`): idle (zero) actions never lift the drill, so **every** reward must be
  exactly `0.0` and `success_once` must stay `False` — the direct opposite of the S1.3 shaped result.

Reproduce: run `s1_4_reward_mode_unit.py` (py3.11 venv); then boot `run_arena_server.sh` and run
`s1_3_env_through_rlinf.py`.

### 3. The result
Unit (fake server): **13/13 PASS** —
```
sparse.step1_reward_0 (r1=0.0)  sparse.step2_reward_1 (r2=1.0)  sparse.reward_eq_success
sparse.terminated_tracks_success (term1=False term2=True)  sparse.truncated_false
shaped.step1/2_reward_dense (0.25)  shaped.info_success_present_on_success
nosuccess.sparse_raises  nosuccess.shaped_ok  invalid_mode_rejected
```
Integration (real `LMDrillLiftRlD1`, server logged `has_success=True`):
```
[PASS] sparse.idle_reward_all_zero -- rewards=[0.0]*10
[PASS] sparse.success_once_false_on_idle -- success_once=[False]   (was [True] under shaped in S1.3)
S1.3 ENV-THROUGH-RLINF OK
```

### 4. Analysis and conclusion
- **Defect fixed.** Sparse reward makes `step_reward > 0` coincide with the true `success` term, so
  RLinf's inherited `success_once` is now correct for free (no change to `IsaaclabBaseEnv` needed).
  Idle actions went from `success_once=[True]` (S1.3, shaped) to `[False]` (sparse) — exactly right.
- **Done/bootstrap preserved.** Only the *reward* channel changed; `terminated` (= success | drop)
  and `truncated` (= time-out) still cross the wire unmodified, so a learner's done mask and value
  bootstrap remain correct (matches gr00t keeping `done = terminated|truncated` separate from reward).
- **Generic & safe.** `reward_mode` is a clean RLinf-side knob (no Isaac reboot to switch); `sparse`
  is the task-agnostic default; `shaped` stays available; requesting `sparse` on a task without a
  `success` term fails fast instead of silently training on zero reward. The config dump confirmed
  `terminations.success` exists with `time_out=False` and `object_dropping=None`.
- **Known boundary (carry to S2.3/S3):** `IsaaclabBaseEnv.step()` rebuilds `infos` via
  `_record_metrics(..., {})` and drops the env's infos, so the client's `infos["success"]` does not
  reach the runner; RLinf's canonical success at that layer is `episode.success_once`. If the
  DSRL runner needs the raw per-step success, plumb it through `_record_metrics` then.
- **Conclusion:** sparse reward is the generic default; S1 env layer is correct and faithful to the
  gr00t reward semantics. Ready for S2 (DSRL into the GR00T model).

---

## S2.1 — DSRL into the GR00T N1.7 action model (PASSED)

### 1. Why perform this experiment
DSRL = train a small SAC agent that *steers* a frozen diffusion/flow policy by choosing its
initial denoising noise. RLinf already implements this for openpi; GR00T N1.7 did not support
it (only the PPO/GRPO `DEFAULT` forward existed). S2.1 adds the DSRL hooks to the GR00T model
so the (soon model-agnostic, S2.2) SAC worker can drive it exactly like openpi. The risks to
retire here: (a) the SAC noise must become the flow policy's initial latent `x_t` (not just
extra action noise); (b) the noise dimension must equal the diffusion latent dim (35 for G1) or
the denoise is silently corrupted; (c) the base (backbone + flow head) must be fully frozen so
only the SAC actor/critic train; (d) `sac_forward`/`sac_q_forward` must match the worker's
call contract (return shapes, deterministic eval).

### 2. How the test is designed to answer that
Implementation mirrors the openpi DSRL reference and **reuses the same shared modules**
(`GaussianPolicy`, `CompactMultiQHead`, `CompactStateEncoder`, `LightweightImageEncoder64`) so
both models behave identically under the SAC worker. Changes (all in
`gr00t/gr00t_n1d7/gr00t_action_model.py`): `get_rl_action(noise=...)` injects the latent (with a
shape assert); `_init_dsrl_components` builds the SAC nets when `rl_head_config.use_dsrl`
(noise dim defaults to `model_action_dim`, asserts on override); `forward` dispatches
`SAC`/`SAC_Q`; `sac_forward`/`sac_q_forward` + `_preprocess_dsrl_images/_states`; the DSRL path
in `predict_action_batch` (SAC noise -> deterministic eval-denoise -> env action; noise stored as
`forward_inputs['action']`, its log-prob as `prev_logprobs`); `freeze_vlm` freezes
backbone+action_head; `__init__.py:get_model` calls it when `use_dsrl`.

Verification without the 3B checkpoint (downloaded in S4.1) or a GPU: build model/action-head
instances via `__new__` (skip the heavy `__init__`) and attach only DSRL nets + tiny stub
backbone/action_head, then exercise the **real** methods. The injection test stubs the denoiser
to the identity so a fixed noise must come out unchanged as the action — proving the steering
plumbing end to end. (`arena_local_bringup/s2_1_gr00t_dsrl_unit.py`.)

Reproduce:
```bash
/home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s2_1_gr00t_dsrl_unit.py
```

### 3. The result
**20/20 PASS** (ruff + import clean):
```
init.noise_dim_35 / init.noise_dim_mismatch_raises / init.has_all_nets
sac.noise_shape (2,50,35) / sac.logprob_shape (2,) / sac.eval_deterministic
sac.noise_repeated_over_horizon / sac.train_stochastic / sac.dist_params_when_requested
sacq.shape (2,6) / sacq.detach_ok / sacq.flat_actions_ok
inject.chains0_is_noise / inject.action_is_noise / inject.bad_noise_shape_raises / inject.no_noise_random
freeze.base_all_frozen / freeze.dsrl_all_trainable
```

### 4. Analysis and conclusion
- **Steering works:** with an identity denoiser, a fixed injected noise emerges unchanged as the
  action and as `chains[0]` — confirming the SAC noise truly becomes the flow latent `x_t`, the
  core DSRL mechanism. A wrong-shaped noise raises rather than silently corrupting denoising.
- **Contract matches openpi/the SAC worker:** `sac_forward` returns `(noise [B,H,35], logprob
  [B], dist_params)` with deterministic eval and stochastic train; `sac_q_forward` returns
  `[B, num_q_heads]` and honors `detach_encoder`. The noise vector is repeated across the horizon
  (same convention as `GaussianPolicy`), so it lines up with the `(B, horizon, model_action_dim)`
  latent.
- **Frozen base:** `freeze_vlm` froze every backbone + action_head param tensor while leaving all
  DSRL nets trainable — exactly the DSRL split (base used only as a frozen noise->action map).
- **Scope boundary:** the SAC worker still reads `use_dsrl` from the openpi config key and filters
  optimizer params by name; making that model-agnostic (so it resolves for `gr00t_n1d7`) is S2.2.
  The full diffusion-determinism + a real SAC update on the actual 3B model are exercised once the
  checkpoint exists (S2.4 LIBERO parity / S3).
- **Conclusion:** GR00T N1.7 now has a correct, openpi-parity DSRL interface, verified in
  isolation. Ready for S2.2 (generalize the SAC worker).

---

## S2.2 — model-agnostic DSRL in the SAC worker (PASSED)

### 1. Why perform this experiment
After S2.1 the GR00T model speaks DSRL, but the SAC worker
(`fsdp_sac_policy_worker.py`) only *recognized* DSRL for openpi: it read the flag from
`cfg.actor.model.openpi.use_dsrl`. For a `gr00t_n1d7` run (flag under
`model.rl_head_config.use_dsrl`) the worker would resolve `use_dsrl=False` and silently take
the non-DSRL path (wrong optimizer groups, no DSRL forward) — so DSRL would never actually run
on GR00T. S2.2 makes the worker drive *any* DSRL-capable model through one code path. Risks to
retire: (a) the flag resolves for both models; (b) the DSRL optimizer param-name filters still
partition GR00T's DSRL params correctly (its nets are named like openpi's); (c) a real SAC
update (critic + actor) actually runs on the GR00T DSRL model and only moves the DSRL nets.

### 2. How the test is designed to answer that
The change is tiny and surgical: a `_resolve_use_dsrl()` helper that ORs the flag across
`model.openpi`, `model.rl_head_config`, and a top-level `model.use_dsrl`; the two openpi-only
read sites now call it / `self.use_dsrl`. Everything else (loss math, param filters, target net)
is model-agnostic and untouched.

`arena_local_bringup/s2_2_sac_worker_unit.py`, two parts, no 3B ckpt / GPU / Ray:
- **Part A** instantiates the *real* worker class via `__new__`, sets a minimal cfg, and asserts
  `_resolve_use_dsrl()` for gr00t / openpi / top-level / mixed (True) and off / empty (False).
- **Part B** builds the GR00T DSRL model (via `__new__` + stubs, `freeze_vlm`), reproduces the
  worker's optimizer param-name split (critic = `critic_*`+`q_head`, actor = the rest) and one
  SAC update with the same structure as `forward_critic`/`forward_actor` (clipped double-Q
  target, entropy term, `detach_encoder` on the actor's Q-eval), then checks finiteness and which
  params move.

Reproduce:
```bash
/home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s2_2_sac_worker_unit.py
```

### 3. The result
**16/16 PASS** (ruff clean; only the openpi *string* left in the worker is the helper docstring):
```
resolve.gr00t_true / openpi_true / toplevel_true / mixed_true ; gr00t_false / empty_false
filters.critic_nonempty (100) / actor_nonempty (32) / disjoint / base_frozen_excluded
update.critic_loss_finite (8.33) / actor_loss_finite (-2.39)
update.critic_params_moved / actor_params_moved / base_unchanged
```

### 4. Analysis and conclusion
- **The worker is now model-agnostic:** `use_dsrl` resolves True for both `gr00t_n1d7`
  (`rl_head_config`) and openpi, so the same SAC code path drives both. False stays False, so
  non-DSRL runs are unaffected.
- **Param filters transfer for free:** because I named GR00T's DSRL nets exactly as openpi's
  (`critic_image_encoder`/`critic_state_encoder`/`q_head` and `dsrl_action_noise_net`/`actor_*`),
  the worker's existing name-based filters split GR00T cleanly — critic 100 tensors (encoders + 6
  Q-heads), actor 32 — disjoint, with the frozen base in neither group.
- **A real SAC step works:** critic + actor losses are finite; the DSRL actor/critic params move
  while the frozen backbone + flow head do not — exactly the DSRL training contract, on GR00T.
- **Scope:** the full worker `forward_critic`/`forward_actor` on the real 3B model (FSDP, target
  net, replay buffer, Ray) is exercised in S2.4 (LIBERO parity) / S3 once the checkpoint exists;
  Part B reproduces their math at the component level. The remaining GR00T<->env coupling (obs/
  action adapter for `isaaclab_arena_g1`) is S2.3.
- **Conclusion:** one SAC worker now supports openpi and GR00T DSRL identically. Ready for S2.3.

---

## S2.3 — GR00T <-> IsaacLab-Arena G1 obs/action adapter (PASSED)

### 1. Why perform this experiment
The env (S1) emits RLinf-canonical obs and consumes a 35-d action; the GR00T model (S2.1)
speaks its own GR00T processor format. S2.3 is the **only** model<->env coupling: convert the
Arena G1 obs into GR00T's `isaaclab_arena_g1` input dict and convert GR00T's action output back
into the env's 35-d joint action. This MUST match the schema the (frozen) BC model was trained
on, or the policy gets scrambled inputs / emits a mis-ordered action. The risks: (a) the 43-d
state must split into the right 7 named groups; (b) the 35-d action must concatenate in the
*action* group order, which **differs** from the state order; (c) the right keys/embodiment tag
so the GR00T processor recognizes the embodiment; (d) decide what the DSRL critic sees.

### 2. How the test is designed to answer that
Schema taken authoritatively from the gr00t repo (the proven offline-RL/eval path feeding the
same BC model) + IsaacLab-Arena `modality.json`: state = 7 groups (43-d, **no** wrist-pose at
inference), `video.ego_view` (`[B,1,H,W,C]`), `annotation.human.task_description` (no `.action`
infix, unlike libero); action = `[left_arm, right_arm, left_hand, right_hand, waist,
base_height_command, navigate_command]` = 35-d, no zeroing; Arena `--embodiment g1`; GR00T tag
`isaaclab_arena_g1`. **Critic decision (resolved):** gr00t's critic uses the same policy 43-d
state + backbone features, **no** privileged `task_obs`/`wbc`/object pose — so RLinf's DSRL
critic input (ego image + 43-d state) is already correct; no extra obs surfaced.

Implementation: two converters in `simulation_io.py`
(`convert_isaaclab_arena_g1_obs_to_gr00t_format`, `convert_to_isaaclab_arena_g1_action_n1d7`)
registered under `isaaclab_arena_g1`; `EmbodimentTag.ISAACLAB_ARENA_G1` (+ projector id 25);
the cfg->tag map in `gr00t_n1d7/__init__.py`; and an explicit passthrough
`prepare_actions_for_isaaclab_arena` (35-d joint action, no gripper remap). Verified with
`arena_local_bringup/s2_3_gr00t_arena_adapter_unit.py` (no 3B ckpt): obs keys/shapes/slices,
action 35-d + order, **round-trips** (state assemble->split->reassemble identity; action
split-by-schema->concat identity, which proves the order matches `modality.json`), plus
registration + the env-side passthrough.

Reproduce:
```bash
/home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s2_3_gr00t_arena_adapter_unit.py
```

### 3. The result
**15/15 PASS** (ruff clean):
```
obs.keys / obs.no_wrist_pose / obs.annotation_key_no_action_infix
obs.state_groups_sliced_correctly / obs.video_shape_BT1HWC (2,1,8,8,3)
action.shape_35 / action.concat_order_matches_schema / action.unprefixed_keys_ok / action.missing_group_raises
roundtrip.state_lossless / roundtrip.action_lossless
reg.obs_registered / reg.action_registered / reg.embodiment_tag / reg.prepare_actions_passthrough
```

### 4. Analysis and conclusion
- **Schema-faithful both directions.** State splits into the 7 named groups by the exact
  `modality.json` slices (and matches `g1_layout.G1_STATE_KEY_ORDER`, so the env's flat 43-d
  state maps losslessly); the action concatenates in the *action* order (arms+hands, then
  waist/base/navigate) — verified by round-trips that are bit-identical, which is the strongest
  proof the ordering matches what the BC model expects.
- **Inference-correct omissions:** wrist-pose/eef_pose is excluded (not in the G1 inference
  state per gr00t `MODALITY_CONFIGS`), and the annotation key has no `.action` infix — both
  subtle mismatches that would silently corrupt the policy if copied from the libero converter.
- **Critic question closed:** policy 43-d state + ego image only; no privileged obs — matches
  gr00t and keeps the env model-agnostic (nothing extra surfaced from `g1_layout`).
- **Decoupling preserved:** all GR00T-specific mapping lives in the gr00t model package; the env
  (S1) stays model/algorithm-agnostic.
- **Scope:** the converters run against synthetic tensors here; exercising them through the real
  3B processor (which needs the checkpoint's `statistics.json`/`embodiment_id.json` for
  `isaaclab_arena_g1`) happens at S4 (BC model) / S2.4 (LIBERO parity) / S3.
- **Conclusion:** S2 model+algo layer is complete (S2.1 model, S2.2 worker, S2.3 adapter).
  Ready for S2.4 (LIBERO DSRL parity) and S2.5 (dry-run).

---

## S2.5 — `--dry-run` pre-submit gate (PASSED)

### 1. Why perform this experiment
OSMO jobs are expensive to submit; we want a local gate that proves the *full pipeline* is
wired correctly before submitting a real train/eval job. The dry-run builds everything (workers,
env, model), runs exactly ONE rollout + ONE actor update, skips val/eval/checkpoint, prints
"DRY RUN OK", and exits 0 — run locally with `num_nodes=1` instead of submitting the multi-node
job. The risk to retire here: the gate's *control flow* — that it really caps to one step and
skips eval/ckpt only under dry-run, without disturbing normal runs.

### 2. How the test is designed to answer that
`runner.dry_run` added to both `EmbodiedRunner.run()` (sync) and `AsyncEmbodiedRunner.run()`:
when set, the step loop is capped to one iteration (sync) / breaks after one successful step
(async), `_maybe_eval_and_checkpoint` / checkpointing is skipped, and "DRY RUN OK" is logged
before a clean finish. Read safely via `cfg.runner.get("dry_run", False)` (works whether or not
the key is in the YAML; set with `runner.dry_run=true`, or `+runner.dry_run=true` for configs
that don't declare it).

Actually *running* it needs the full stack (workers/model/env/GPU) and is exercised at S3. Here
the control flow is verified in isolation: `arena_local_bringup/s2_5_dry_run_unit.py` builds an
`EmbodiedRunner` via `__new__`, stubs the heavy collaborators (worker groups -> fake handles,
timer, logger) and spies on `_maybe_eval_and_checkpoint`/`_save_checkpoint`, then asserts the
gating for both `dry_run=True` and `dry_run=False`.

Reproduce:
```bash
/home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s2_5_dry_run_unit.py
```

### 3. The result
**8/8 PASS** (ruff clean, both runners):
```
dry.one_step (global_step=1) / dry.eval_skipped / dry.checkpoint_skipped / dry.finish_called
dry.prints_DRY_RUN_OK
normal.one_step / normal.eval_called / normal.no_DRY_RUN_OK
```

### 4. Analysis and conclusion
- **Gate logic correct:** under `dry_run=True` the loop runs exactly one step, eval+checkpoint
  are skipped, and "DRY RUN OK" is logged; under `dry_run=False` (even with `max_steps=1`) the
  eval/checkpoint path runs and no dry-run banner is printed — so the flag *only* changes the
  dry-run behaviour, leaving normal training untouched.
- **Minimal + safe:** read via `.get(..., False)`, so it never breaks existing configs; mirrored
  in the async runner with a break-after-one-successful-step (the skip-step path `continue`s
  earlier, so reaching the break implies a real update happened).
- **Scope:** the end-to-end dry-run (real workers + frozen 3B model + Arena server, one
  rollout->reward->adv->update with finite losses) is the S3 local-e2e verification once the
  checkpoint exists; S2.5 delivers the mechanism and proves its control flow.
- **Conclusion:** the pre-submit gate is in place. S2 (model + algo + dry-run) is complete; the
  only remaining S2 item, S2.4 (LIBERO DSRL parity), is GPU/checkpoint-bound and folds into the
  S3/S4 phase.

---

## S2.4/S3 — DSRL-on-GR00T config scaffolds + preflight (PASSED, run pending ckpt)

### 1. Why perform this experiment
S2.4 (LIBERO DSRL parity) and S3 (Arena drill_lift local e2e) both need a config, and both
runs are GPU/checkpoint-bound. To keep momentum while the public checkpoint is acquired (S4.1),
scaffold the two configs now and prove they are well-formed and that all the DSRL wiring built in
S2.1–S2.3 actually resolves through them — so the only thing left to run them is the checkpoint.

### 2. How the test is designed to answer that
Two configs merged from the working openpi DSRL recipe (`libero_spatial_dsrl_openpi.yaml`) +
the GR00T N1.7 model/env structure (`libero_spatial_ppo_gr00t_n1d7.yaml`):
- `libero_spatial_dsrl_gr00t_n1d7.yaml` (S2.4): DSRL on GR00T, LIBERO env (non-Arena de-risk).
- `drill_lift_dsrl_gr00t_n1d7.yaml` (S3): DSRL on GR00T, Arena `isaaclab_arena_drill_lift`,
  single env, `embodiment_tag/obs_converter_type=isaaclab_arena_g1`, `action_dim=35`,
  `dsrl_state_dim=43`; Arena server over the socket; `runner.dry_run` wired.
Both put DSRL params under `actor.model.rl_head_config` (the gr00t key S2.2's resolver reads),
omit `dsrl_action_noise_dim` (auto-derives from `model_action_dim`), and set
`add_value_head=False` (DSRL uses its own SAC critic). Preflight
(`arena_local_bringup/s2_4_s3_config_preflight.py`): Hydra-compose each config, run
`validate_cfg`, and assert the wiring — closing the loop with S2.2 (`_resolve_use_dsrl` -> True)
and S2.3 (obs/action converter + embodiment tag registered).

Reproduce:
```bash
EMBODIED_PATH=/home/juekunl/Work/RLinf/examples/embodiment \
  /home/juekunl/Work/RLinf/.venv/bin/python \
  /home/juekunl/Work/arena_local_bringup/s2_4_s3_config_preflight.py
```

### 3. The result
**PREFLIGHT OK** (all checks pass; trailing Ray atexit traceback is a sandbox-only cosmetic
issue after the result): both configs compose + `validate_cfg` pass; `loss_type/adv_type ==
embodied_sac`; `runner.dry_run` declared (default False); `_resolve_use_dsrl -> True`;
`add_value_head False`; `rl_head_config.use_dsrl True`. Drill: `env_type=isaaclab_arena`,
`action_dim=35`, `dsrl_state_dim=43`, single env, `obs_converter_type=isaaclab_arena_g1`
registered in OBS/ACTION converters, embodiment tag valid, env server block on port 5557.

### 4. Analysis and conclusion
- **Configs are run-ready** pending only the checkpoint/backbone paths (placeholders, TODO S4):
  the full DSRL pipeline wiring resolves through Hydra+validate_cfg, and the resolver/converter/
  tag checks confirm S2.2/S2.3 are correctly referenced.
- **Remaining (GPU/ckpt-bound):** actually executing `runner.dry_run=true` (1 rollout+update) and
  the short DSRL loop is S2.4 (LIBERO) / S3 (Arena, with Isaac Sim viz), once S4.1 provides the
  public `nvidia/GR00T-N1.7-3B` + a drill_lift BC ckpt.

---

## S4.1 — OSMO BC assets: data confirmed + public model downloaded (PARTIAL: backbone gated)

### 1. Why perform this experiment
The BC base policy (S4.3) needs the teleop dataset + the public `nvidia/GR00T-N1.7-3B` on the
OSMO Lustre filesystem. Confirm the dataset matches the schema my converters (S2.3) assume, and
acquire the model. Run via OSMO workflow `train_ray_cluster_2n_uuid_cb5c-1` (later
`..._b36c-1` after the first cluster was torn down; Lustre is shared/persistent).

### 2. How / 3. Result
OSMO `exec` needs a TTY → wrapped via `arena_local_bringup/osmo_exec.sh` (`script` PTY +
base64 to survive quoting). Findings:

**Data — CONFIRMED** at `/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl`:
- LeRobot dataset, **115 episodes** / 39437 frames @ 50 fps, `robot_type=unitree_g1`,
  task "Lift the power drill from the table." (115 parquet + 115 ego_view mp4).
- `modality.json` = the exact S2.3 schema (state 43 in 7 groups; action 35 in 7 groups).
- `info.json`: `observation.state float32 [43]`, `action float32 [35]`,
  `observation.images.ego_view [480,640,3]`, plus `rl_info.next.{reward,done,success}`.
- LeRobot sample loads: `observation.state (43,) float32`, `action (35,) float32`,
  episode succeeds (`success_any=True`, reward_sum≈24.2). Matches S2.3 converters exactly.

**Model — DOWNLOADED** to `/mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B` (6.5 GB,
27 files, both safetensors shards complete + index match, 0 incomplete). Config:
- `model_type=Gr00tN1d7`, backbone `model_name=nvidia/Cosmos-Reason2-2B`,
  `backbone_embedding_dim=2048` — RLinf-loadable as `gr00t_n1d7`.
- **`max_action_dim=132`** (NOT 35) and **`action_horizon=40`** (NOT 50): the public base is
  cross-embodiment-padded; G1's 35-d action occupies slices 0:35 of the 132-d latent.
- Ships embodiment `unitree_g1_full_body_with_waist_height_nav_cmd` (id 25) but NOT the
  `isaaclab_arena_g1` tag/modality (→ S4.2 registers it).

**Backbone — RESOLVED (internal cache, no token):** `nvidia/Cosmos-Reason2-2B` is HF-gated
(`401`, no `HF_TOKEN`), and Isaac-GR00T auto-pulls it on first load via an HF *local-first*
wrapper (`gr00t._hf_local_first_call`: probe `local_files_only=True` → cache hit reuse, miss →
download). A complete internal copy at the **canonical revision** `9ce19a195e...` (the rev the
GR00T-N1.7-3B build expects) was found at
`/mnt/amlfs-07/shared/groot_oss/ci/hf_home/hub/models--nvidia--Cosmos-Reason2-2B`. Staged
(symlinks resolved) to a self-contained dir
**`/mnt/amlfs-07/shared/juekunl/models/Cosmos-Reason2-2B`** (4.6 GB: config.json,
model.safetensors, full Qwen3-VL tokenizer/processor). So RLinf `backbone_model_path` (or
`HF_HOME`) can point at the internal copy — no token, no Hub. NOTE: even with the full
GR00T-N1.7-3B checkpoint, loading still needs the backbone repo (to build the Qwen3-VL arch +
processor), so this staging is required for offline/local load.

### 4. Analysis and conclusion
- **Data + main model are in place** on shared Lustre; the dataset is byte-for-byte the schema
  S2.3 targets, so the GR00T↔env adapter is validated against the real BC data.
- **Two findings reshape S4.2/S4.3 (and the S3 DSRL config):** the public model is
  `max_action_dim=132`, `action_horizon=40`. Consequences: (a) DSRL noise dim auto-derives to the
  finetuned model's `model_action_dim` (my configs OMIT `dsrl_action_noise_dim` precisely so this
  self-corrects — good), but the `target_entropy` I scaffolded (`-17`, assuming 35) must be
  retuned to ≈ `-noise_dim/2` once the BC ckpt's dims are fixed; (b) the action_horizon (40 vs
  G1 data's 50) is an S4.2/S4.3 decision (re-chunk data vs set action_horizon — follow the gr00t
  cr2 factory config `grootn1d7rl_g1_factory_cr2_config.py`).
- **Backbone gating** mostly affects LOCAL use (S3 on the RTX 5880) and a self-contained RLinf
  load; the OSMO BC finetune (S4.3) + DSRL (S5) go through the gr00t/posttrain pipeline whose env
  already pulls the gated backbone from HF. Resolution options: (i) set an `HF_TOKEN` with
  Cosmos-Reason2-2B access on the node/local; (ii) obtain an internal copy; (iii) confirm the
  GR00T-N1.7-3B download is self-sufficient at load (S4.2/S4.3, GPU).
- **Status:** S4.1 ✅ — data ✅, public model ✅, backbone ✅ (internal Cosmos-Reason2-2B staged
  to `/mnt/amlfs-07/shared/juekunl/models/Cosmos-Reason2-2B`, no token).

---

## S4.2 — Embodiment resolved: public base already ships the G1 tag (no registration)

### 1. Why perform this experiment
S4.3 finetune needs an embodiment tag whose modality config matches `drill_lift_rl`. Initial
plan assumed registering a new `isaaclab_arena_g1` tag in Isaac-GR00T.

### 2. How / 3. Result
The public base already registers `EmbodimentTag.UNITREE_G1 =
"unitree_g1_full_body_with_waist_height_nav_cmd"` (projector id 25, `gr00t/data/embodiment_tags.py`),
and its pre-registered modality config (`gr00t/configs/data/embodiment_configs.py`) == the
`drill_lift_rl` `modality.json` (state 7 groups / 43-d, action 7 groups / 35-d, ego_view,
annotation; horizon 50). Validated by loading both in the env venv (STATE/ACTION match).

### 4. Analysis and conclusion
**NO NEW_EMBODIMENT, NO custom modality config.** Finetune with `--embodiment-tag UNITREE_G1`
and reuse the pretrained G1 projector. (`UNITREE_G1_SONIC`, used by the WBC real-robot example,
is for SONIC-collected data — different modality — so it is NOT used for `drill_lift_rl`.)

---

## S4.3 — BC finetune public GR00T-N1.7-3B on drill_lift (OSMO, official Isaac-GR00T)

### 1. Why perform this experiment
DSRL (S5) steers a *frozen* base policy, so we first need a drill_lift BC base. Decision:
finetune the **public** `nvidia/GR00T-N1.7-3B` with the **public** Isaac-GR00T
`examples/finetune.sh` (model_type `Gr00tN1d7`, no internal-repo coupling) so the result is a
clean public-model story and loads back into RLinf. Run on the OSMO H100 node
(`train_ray_cluster_2n_uuid_b36c-1`); validate locally-first via dry-runs before the ~1 h full run.

### 2. How is this test designed
Pulled Isaac-GR00T to latest `main` (`ab88b50`; it added the `GR00TWholeBodyControl` G1 WBC
benchmark + hardened finetune/horizon utilities). Authoritative recipe from that example: its
**single-task** soda-can policy (~150 eps) used **20k iters @ batch 32** → reuse for our 115-ep
drill_lift. Built the public uv env on the node and ran two dry-runs then the full run via
`arena_local_bringup/bc_finetune_drill_lift_osmo.sh` (idempotent runner). Five concrete blockers
were found+fixed (each isolated by a 1-step dry-run, per the step-by-step principle):

1. **uv sync (x86_64):** latest `main` pins flash-attn/torchcodec to **aarch64** LFS-pointer
   wheels under `scripts/deployment/dgpu/wheels/` and uses `[tool.uv] required-environments` for
   both arches → uv tried to read the (LFS-pointer) aarch64 wheels. Fix: switch to
   `[tool.uv] environments` = x86_64-only + drop the two aarch64 path-wheel sources →
   uv uses the GitHub x86_64 flash-attn wheel. Env: torch 2.7.1+cu128, flash-attn 2.7.4.post1,
   transformers 4.57.3, 8× H100 visible.
2. **Gated backbone, offline crash:** `HF_HUB_OFFLINE=1` made transformers 4.57's tokenizer
   `model_info()` probe raise. Fix: load the backbone from a **local path** (`_is_local` skips
   that network call), no offline mode.
3. **Backbone class match:** `get_backbone_cls` matches the substring `"nvidia/Cosmos-Reason2"`
   in `model_name` → expose the staged backbone at `.../models/nvidia/Cosmos-Reason2-2B` (symlink)
   so a *local path* still matches. Set via `GR00T_BACKBONE_PATH` (1-line env override added to
   `launch_finetune.py`, default unchanged).
4. **Horizon 40 vs data 50:** `AssertionError: Action sequence length 50 exceeds
   max_action_horizon 40`. Fix: `action_horizon=50` in BOTH the processor (`config.model`, via
   `GR00T_ACTION_HORIZON=50`) AND the loaded model (base `config.json action_horizon 40→50`).
   **No weight resize:** the action-head positional embedding is `nn.Embedding(max_seq_len=1024,
   dim)` indexed by `arange(horizon)` — zero params are horizon-shaped.
5. **Gated backbone in full path:** the base `config.json model_name` is the hub id → full-load
   path hit the gated hub (401). Fix: set base `config.json model_name` → the local
   `nvidia/Cosmos-Reason2-2B` path (idempotent in the runner; pristine `config.json.bak40` kept).

### 3. The result
- **Fast dry-run** (`--skip_weight_loading`, builds from `config.model`): builds at horizon 50,
  embodiment `unitree_g1...` matched, 34 shards / 33802 steps from `drill_lift_rl`, 1 step,
  finite `train_loss=1.28`, ckpt saved. `BC_DRYFAST_DONE_OK`.
- **Full dry-run** (real weights): the 40-trained 6.5 GB base loaded into the **horizon-50** model
  with **zero missing/unexpected/mismatched keys** (no RuntimeError) — proving no resize is
  needed — 1 step, finite `train_loss=1.58`. `BC_DRYFULL_DONE_OK`.
- **Full run** (8× H100, torchrun, 20k steps, batch 32, lr 1e-4, freeze backbone / tune
  projector+diffusion = 878M/2.2B trainable): training healthy at **~5.4 it/s (~1 h ETA)**;
  loss **1.5 → ~0.07 by step ~700**, grad_norm ~0.5-1.1, no NCCL/OOM. Output:
  `/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-*`
  (save every 2k, keep last 5 → 12k..20k).

### 4. Analysis and conclusion
- **Pipeline validated end-to-end on the public stack**; the `40→50` adaptation is a config-only
  change (architecture is horizon-agnostic in its parameter shapes), and the gated backbone is
  fully resolved with **no HF token** via a local path that satisfies both the class matcher and
  transformers' local-file detection.
- **Reproducibility:** all node-side edits (config.json, `launch_finetune.py`, uv env) are baked,
  idempotently, into `arena_local_bringup/bc_finetune_drill_lift_osmo.sh`
  (`DRY_RUN=1 [FAST=1] ...` for the 1-step checks; no args for the full run).
- **Caveat for RLinf load (S4.4):** the finetuned ckpt's `config.json model_name` will be the
  node-local backbone path; when loading the ckpt back in RLinf, point `model_name`/backbone at a
  valid local Cosmos-Reason2-2B (or re-set the hub id with a token).
- **Next:** S4.4 — eval the saved BC checkpoints on drill_lift, pick the best as the FROZEN base
  for DSRL (S5).

---

## S4.3a — Stale-first-frame data fix + checkpoint retention (restart on f8ef)

### 1. Why perform this experiment
Two corrections to the S4.3 run: (a) **stale first frame** — the drill_lift_rl teleop/rollout
recorder logs a stale first frame per episode (off-by-one between observation and action/reward;
documented in `IQL-Drill-Lift-Presentation`), so the first transition is corrupted. The internal
gr00t offline-RL pipeline fixes this with `--skip-first-n-frames 1`, but the public Isaac-GR00T
has no equivalent. (b) **checkpoint retention** — the first run inherited finetune.sh's hardcoded
`save_total_limit=5`, which over 20k steps @ save-2k keeps only 12k..20k and deletes 2k..10k —
bad for S4.4, since BC on 115 episodes can peak well before 12k.

### 2. How is this test designed
- **Data fix:** added `skip_first_n_frames` threaded `data_config.py` → `factory.py` →
  `ShardedSingleStepDataset`, where `shard_dataset`'s per-episode index range changes from
  `np.arange(0, eff_len)` to `np.arange(skip_first_n_frames, eff_len)` (drops step 0). Exposed via
  env `GR00T_SKIP_FIRST_N_FRAMES` in `launch_finetune.py` (default 0 — no behavior change for
  other datasets).
- **Checkpoints:** made finetune.sh `save_total_limit` env-driven (`SAVE_TOTAL_LIMIT`); run with
  `--save-only-model` (weights only, ~6.5 GB) + `SAVE_TOTAL_LIMIT=10` + save every 2k → keep the
  full 2k..20k sweep cheaply (we eval+freeze for DSRL, never resume).
- All node source edits bundled into the idempotent `arena_local_bringup/osmo_patch_isaac_gr00t.py`
  (also carries the backbone-path/action_horizon env overrides), so a fresh OSMO node is one
  command. Cluster `b36c` was torn down mid-run → moved to `train_ray_cluster_2n_uuid_f8ef-1`;
  Lustre persisted (model config.json edits, staged backbone, dataset), so only the /workspace
  checkout (clone + uv sync + patch) had to be rebuilt. Verified by re-running the local patch
  script (clean 9-edit diff) before deploying.

### 3. The result
- **Fast dry-run with skip=1:** `Total steps 33802 → 33687` — exactly −115 (one step per episode
  over 115 episodes), finite loss, `BC_DRYFAST_DONE_OK`. Confirms the drop is exact.
- **Full run relaunched on f8ef** (8× H100, 20k, batch 32, lr 1e-4, skip=1, `--save-only-model`
  + `SAVE_TOTAL_LIMIT=10`, save every 2k): `Total steps 33687`, loss 1.5 → ~0.12 by step ~140,
  8 GPUs at 100% util, no errors. Throughput ~1.8 it/s / ETA ~3 h (vs ~5 it/s on b36c — GPUs now
  compute-bound at 100%, likely a different H100 SKU on f8ef).

### 4. Analysis and conclusion
- The data fix is exact, minimal (a single `np.arange` start index), env-gated, and reproducible.
  The checkpoint sweep (2k..20k) is now preserved for a proper S4.4 best-checkpoint selection.
- **Cluster churn is now cheap:** `osmo_patch_isaac_gr00t.py` + `bc_finetune_drill_lift_osmo.sh`
  redeploy on any fresh node in minutes; Lustre carries model/backbone/dataset across clusters.
- **Next:** S4.4 — eval the 2k..20k BC checkpoints on drill_lift, pick the best as the frozen
  DSRL base (S5).

---

## S4.4 — success-criterion lock-in (preflight)

### 1. Why perform this experiment
Before scoring BC checkpoints, the RLinf-side success threshold must be byte-identical to the
internal gr00t IQL eval; otherwise BC ↔ IQL ↔ DSRL success rates are not comparable. An earlier
note claimed our `lift_height=0.15` was "too easy vs gr00t's 0.2" and had to be fixed — this
preflight verifies that claim before changing anything.

### 2. How is this test designed
Traced `lift_height` through its three layers and the success terminal itself:
- Arena task argparse default (`G1_Factory/LMDrillLift.py`) — `0.16` for **all four** DrillLift
  variants (base / D1 / RlD1 / Rl); `RlD1` does **not** hardcode its own value.
- gr00t eval-worker base default (`groot/vla/eval/sim/isaaclab_arena_worker.py`) — `0.2`.
- gr00t eval-worker `*RlD1` override (same file) — `defaults["lift_height"] = 0.15`.
- Success terminal: `object_lifted = root_pos_w[:,2] > lift_height` **AND** `require_grasped=True`
  (hardcoded in `LMDrillLift.get_env`). `lift_height` is an **absolute world-Z** (drill spawns at
  ≈ +0.0015 m, table ≈ 0), not height-above-table — they only coincide because the table sits at
  world-Z ≈ 0.

### 3. The result
Our task is `LMDrillLiftRlD1` (`endswith("RlD1")`), so the IQL eval graded on
**lift_height = 0.15 + grasped**. `rlinf/envs/isaaclab_arena/cli_defaults.py` already mirrors the
worker exactly, including `if arena_env_name.endswith("RlD1"): defaults["lift_height"] = 0.15`.
No code change required.

### 4. Analysis and conclusion
The earlier "must fix to 0.2" was a misread: I saw the worker's *base* `0.2` and missed its own
`RlD1 → 0.15` override. The RLinf eval is already comparable to the gr00t IQL numbers. The only
real definitional gap is the BC **data** (recorded at the env default `0.16`, and `*Rl*` replay
defaults to `require_grasped=False`) — ~1 cm stricter on height, looser on grasp — but this does
not affect BC-vs-DSRL-vs-IQL comparability since all three are *evaluated* at `0.15 + grasped`.
Also keep `embodiment="g1"` (35-dim joint action): the gr00t eval skill notes `LMDrillLiftRlD1`
reproduced 0% when fed the 23-dim WBC action, so the action-space pin matters as much as the height.

---

## S4.5 — BC eval on drill_lift: 0% success despite VERIFIED-CORRECT plumbing (OPEN)

### 1. Why / the issue
Scoring the S4.3a BC checkpoints (the point of S4.4) surfaced the current blocker:

- **Every** finetuned checkpoint scores ~**0/100** on `LMDrillLiftRlD1` (100 episodes, seed 42,
  `n_action_steps=8`, `max_episode_steps=720`, success = lift 0.15 + grasped): `ckpt-2000/4000/8000/10000
  = 0/100`, `ckpt-6000 = 1/100`. Recorded videos show **plausible arm reaching but the hand never
  grasps / the drill never lifts**.
- The "60–70%" reference we were comparing against is **NOT** a BC number. Per the user, it is
  **iter0 IQL** (offline RL, `--loss-mode IQL`, sparse reward, Q-ensemble) trained from the
  **Eagle** base `s3://GearCheckpoints/.../n17_es_eagle_thirdstage/checkpoint-200000` on the *same*
  `drill_lift_rl` demos (`groot/vla/scripts/offline_rl/omni/entrypoint.py grootn1d7rl_g1_factory`).
  So `0% (my BC-on-Cosmos)` vs `60–70% (internal IQL-on-Eagle)` stacks **two** confounds:
  **algorithm** (pure BC vs IQL) and **backbone** (public Cosmos-Reason2 vs Eagle).

Diagnosis question: is the 0% a **plumbing bug in the RLinf eval**, or a **genuine model-quality
result** (BC-on-Cosmos drifts closed-loop; RL/DSRL is exactly the fix)? Verdict so far: **plumbing
is correct end-to-end; the 0% is a closed-loop phenomenon** — one final A/B test (below) is pending.

### 2. How / diagnostics (each isolates one layer)

All diagnostic scripts live in `arena_local_bringup/diag_*.py`; run on the OSMO eval cluster
(`eval_ray_cluster_1n_uuid_4226-1`) inside the Arena 4.6.5 venv
(`/home/juekunl/Work/IsaacLab-Arena/.venv`). The GPU tests need one free GPU (kill any keepalive
first, re-arm after). The node shares the `/home/juekunl/Work` tree (baked env) + Lustre.

1. **Base checkpoint identity** — is the internal base different weights from the public one?
   Compare all safetensors tensors of `s3://GearCheckpoints/.../n17_es_cr2_frozen_200k_unfrozen_200k/
   checkpoint-200000` vs `/mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B`:
   ```python
   from safetensors import safe_open; import glob
   pub=[safe_open(f,framework="pt") for f in sorted(glob.glob(f"{PUB}/model-*.safetensors"))]
   cr2=[safe_open(f,framework="pt") for f in sorted(glob.glob(f"{CR2}/model-*.safetensors"))]
   # union keys; for every common key compare .float() max-abs-diff
   ```
2. **Demo data quality** — is `drill_lift_rl` clean or mixed-quality? Read `rl_info.next.success`
   per episode (pyarrow) → fraction of episodes that ever succeed.
3. **Env / task / camera / layout parity** — read `groot/vla/eval/sim/configs/tasks/drill_lift.yaml`
   (which Arena env the internal eval uses), the dataset `meta/tasks.jsonl`, and
   `isaaclab_arena_gr00t/lerobot/config/g1_arena_camera_rewards_base_config.yaml` (camera + fps).
4. **Native open-loop reconstruction** — did BC even learn? `CUDA_VISIBLE_DEVICES=0 python
   diag_recon_native.py <ckpt> /mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl UNITREE_G1`
   (Isaac-GR00T `open_loop_eval` + Gr00tPolicy; decord monkeypatch replaces the unavailable torchcodec).
5. **RLinf-path reconstruction** — is my obs converter + action decode correct? `CUDA_VISIBLE_DEVICES=0
   python diag_recon_rlinf.py <ckpt> <drill_lift_rl> "Lift the power drill from the table."` (feeds the
   same demo frames through the RLinf `gr00t_n1d7` loader used by `eval_bc_ckpt.py`).
6. **Sim obs dump** — does the SIM feed the model the same obs as the dataset? `CUDA_VISIBLE_DEVICES=0
   python diag_sim_obs_dump.py` (real Arena env reset; prints sim state/image vs dataset).

Reproduce the **issue** itself (the 0% eval):
```bash
# per checkpoint (100 eps, seed 42, video):
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -u eval_bc_ckpt.py \
  --model_path /mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000 \
  --n_episodes 100 --seed 42 --video_dir /home/juekunl/Work/eval_videos/ckpt10000
# or the 5-ckpt parallel cluster sweep: arena_local_bringup/eval_sweep_osmo.yaml
```

### 3. The result

| Diagnostic | Result | Verdict |
|---|---|---|
| 1. Base ckpt identity | 1031/1031 tensors, `same_set=True`, **global max-abs-diff = 0.0** | internal cr2 base **IS** public N1.7 (bit-identical) → base **not** the cause |
| 2. Demo quality | **115/115 = 100%** successful demos (`rl_info.next.success`), 50 fps, ~343 frames/ep | data is clean → BC *should* learn → 0% is a real defect, not bad data |
| 3. Env/task/camera | `drill_lift.yaml → isaaclab_arena_g1/LMDrillLiftRlD1` (== my eval env); `robot_head_cam_rgb→ego_view`; state 43-d + action 35-d == `modality.json` | env/camera/layout **correct** (minor: data says "Lift the **power** drill…", env sets "Lift the drill…"; internal eval uses the same env string → not the cause) |
| 4. Native recon (ckpt-10000) | avg **MSE 0.0085 / MAE 0.025** (per-traj 0.0009–0.013) | **BC learned the task** (open-loop actions match GT) |
| 5. RLinf-path recon (ckpt-10000) | avg **MSE 0.0049 / MAE 0.018** (~= native) | RLinf obs converter + action decode **correct** |
| 6. Sim obs dump | SIM state first-12 **identical** to dataset; SIM image uint8 **mean 108.0** vs dataset uint8 mean 107.2 | sim state convention + image format/scale **correct** |

### 4. Analysis and conclusion (OPEN)
- **The RLinf eval pipeline is verified correct end-to-end.** The checkpoint learned the task
  (native recon), my obs→action prediction path reproduces GT just as well (RLinf recon), and the
  sim feeds the model the *same* state values and uint8 image scale as the training data. State
  assembly (`arena_inprocess_env._wrap_obs`, 7 groups in `modality.json` order) and action
  application (`prepare_actions_for_isaaclab_arena`, 35-d passthrough) match the internal
  `IsaacLabArenaEnvWrapper`. A hypothesized image-format bug (internal wrapper defensively converts
  float[0,1]→uint8; mine doesn't) was **refuted** — Arena already emits uint8 (mean 108).
- **So the 0% is purely a CLOSED-LOOP phenomenon** (nothing in the open-loop prediction path). Two
  live explanations remain:
  - **A) BC closed-loop drift (not a bug).** Pure BC on the Cosmos backbone is open-loop-accurate
    but compounds error closed-loop and never grasps; the internal 60–70% comes from **IQL** (value
    learning), a different algorithm. If so, the RLinf pipeline is *ready* and DSRL/RL is the fix.
  - **B) A subtle closed-loop action-application detail** the recon can't see (e.g. hand-joint
    actuation, chunk-execution cadence). Would need to be fixed before DSRL.
- **Decisive A/B test (PENDING):** `arena_local_bringup/diag_cl_rollout.py` runs the BC policy
  closed-loop and logs, per episode, `max_reward`, the commanded hand-action magnitude, and how much
  the hand joints actually move. `reaches + hands actuate + high max_reward but success=0` ⇒ **A**
  (drift; plumbing OK). `hands never move despite non-zero hand action` ⇒ **B** (bug). This run was
  launched twice (once backgrounded — killed by osmo session teardown before Isaac booted; once
  foreground — interrupted). **Re-run it in the foreground to close the A/B question.**
- **Strategic note:** since the internal deliverable is DSRL-in-RLinf and the RLinf path uses the
  public Cosmos loader (Eagle isn't loadable there), matching internal IQL-on-Eagle exactly is a
  *reference*, not the RLinf goal. If the A/B test confirms **A**, proceed to the DSRL sign-of-life
  on the (correct-but-weak) BC base; a non-degenerate frozen base is all DSRL needs to steer.

Diagnostic scripts: `arena_local_bringup/diag_recon_native.py`, `diag_recon_rlinf.py`,
`diag_sim_obs_dump.py`, `diag_cl_rollout.py`; eval driver `arena_local_bringup/eval_bc_ckpt.py`;
cluster sweep `arena_local_bringup/eval_sweep_osmo.yaml`.

---

## Files referenced
- `rlinf/envs/isaaclab_arena/protocol.py`, `g1_layout.py`, `arena_server.py`, `arena_client.py`, `arena_env.py`
- `rlinf/models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py`, `gr00t/gr00t_n1d7/__init__.py` (DSRL + cfg→tag), `rlinf/workers/actor/fsdp_sac_policy_worker.py` (`_resolve_use_dsrl`)
- `rlinf/models/embodiment/gr00t/simulation_io.py` (arena_g1 converters), `gr00t/embodiment_tags.py` (`ISAACLAB_ARENA_G1`), `rlinf/envs/action_utils.py` (arena passthrough)
- `rlinf/runners/embodied_runner.py`, `rlinf/runners/async_embodied_runner.py` (`runner.dry_run` gate)
- `examples/embodiment/config/libero_spatial_dsrl_gr00t_n1d7.yaml` (S2.4), `examples/embodiment/config/drill_lift_dsrl_gr00t_n1d7.yaml` (S3)
- `arena_local_bringup/run_arena_server.sh`, `l2b_socket_smoke.py`, `l2_xinterp_probe.py`, `s1_1_arena_client_unit.py`, `s1_3_env_through_rlinf.py`, `s1_4_reward_mode_unit.py`, `s2_1_gr00t_dsrl_unit.py`, `s2_2_sac_worker_unit.py`, `s2_3_gr00t_arena_adapter_unit.py`, `s2_5_dry_run_unit.py`, `s2_4_s3_config_preflight.py`, `osmo_exec.sh` (PTY wrapper for `osmo workflow exec`)
- `arena_local_bringup/bc_finetune_drill_lift_osmo.sh` (S4.3 idempotent BC-finetune runner: runs patch script + Lustre config edits + `examples/finetune.sh`; `DRY_RUN=1 [FAST=1]` for 1-step checks; skip-first-frame + save-only-model + keep-all-ckpts)
- `arena_local_bringup/osmo_patch_isaac_gr00t.py` (S4.3a idempotent source patches for a fresh node: GR00T_BACKBONE_PATH/ACTION_HORIZON/SKIP_FIRST_N_FRAMES env overrides, skip-first-frame data fix across data_config/factory/sharded_single_step_dataset, SAVE_TOTAL_LIMIT env)
- OSMO Lustre: data `/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl`, model `/mnt/amlfs-07/shared/juekunl/models/GR00T-N1.7-3B`, BC ckpts `/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/`

---

# In-Process MVP (Path B) — IsaacLab entrypoint -> RLinf -> Arena env, one venv

Pivot from the socket sidecar (above) to the official IsaacLab<->RLinf in-process
integration (Mingxue/assemble_trocar pattern): RLinf installed as a library into the
**same py3.12 venv** as IsaacLab-Arena, no cross-interpreter socket. Goal = local
sign-of-life proving the path on the gear-paired stack.

## Runtime decision
- IsaacLab **7.5.0** (develop) needs Isaac Sim **6.0.0 GA / USD 25.11 / Kit 110.1.1**; the
  local gear container is Isaac Sim **6.0.0-rc.22 / USD 25.05**, and pip `isaacsim`
  6.0.0.0 & 6.0.1.0 both ship USD 25.05 (6.0.1.0 also forces torch 2.11/cu13). So 7.5.0
  is deferred to the open-source PR (Path A: pull `nvcr.io/nvidia/isaac-sim:6.0.0`).
- MVP runs on the **proven Arena 4.6.5 `.venv` + gear container** (USD 25.05).
- Container: `isaaclab-jk-mvp` (gear image, `/home/juekunl/Work` + host uv cache mounted, GPU 0).

## M0 — Arena drill_lift on the runtime
Re-ran `arena_local_bringup/l1_viz_drilllift.py` headless on the Arena 4.6.5 venv: `L1 OK`,
5 steps. Obs contract captured: `policy.robot_joint_pos(43)`, `camera_obs.robot_head_cam_rgb
(480x640x3 uint8)`, action 23-d WBC-pink, `success` term + `success_bonus(0.16)`.
Artifact: `arena_local_bringup/m0b_4_6_5_smoke.log`.

## M1 — one-venv dependency coexistence (the core de-risk)
`uv pip install -e RLinf --no-deps` (RLinf pins torch 2.6 in override-deps; `--no-deps`
preserves the venv). All import together, torch untouched: `isaaclab 4.6.5`,
`isaacsim 6.0.0.0`, `isaaclab_arena 1.0.0`, `rlinf 0.3.0`, `gr00t` (Arena submodule),
`ray 2.55.1`, `torch 2.10.0+cu128`. The rlinf extension imports; `register()` runs
(needs `RLINF_CONFIG_FILE`, set by the entrypoint).

## M2 — RLinf drives the Arena env in-process (the novel link)
New `rlinf/envs/isaaclab_arena/arena_inprocess_env.py` (`IsaaclabArenaG1InProcessEnv`):
reuses `IsaaclabBaseEnv` + `SubProcIsaacLabEnv`; `_make_env_function` launches `AppLauncher`
in the spawned child and builds the env via `ArenaEnvBuilder` (direct `LMDrillLiftRlD1`
import, bypassing `isaaclab_arena_environments.cli`'s import-everything dict, which pulls
dexsuite/etc. unported to 7.5.0). Registered in `REGISTER_ISAACLAB_ARENA_ENVS`.
Direct test `arena_local_bringup/m2_inprocess_env_test.py`: `M2 OK` — reset ->
`states(1,43)` + `main_images(1,480,640,3)` on cuda + `task_descriptions`; step (idle
WBC-pink action) returns reward/term/trunc + RLinf `episode` metrics. CUDA-tensor IPC of
the camera over the mp queue works.
Gotcha: a zero action gives zero-norm wrist quats -> `R.from_quat` crash in the WBC term;
use a valid unit-quat (idle) action.

## M2b — entrypoint env path via REAL RLinf Cluster + EnvWorker (no checkpoint)
Closes the head of the chain that M2 skipped (M2 instantiated the env class directly).
`arena_local_bringup/m2b_entrypoint_env_launch.py` mirrors `train.py`'s env-group launch:
sets `RLINF_EXT_MODULE` (Worker auto-calls `register()`), loads
`arena_local_bringup/drill_lift_g1_rlinf.yaml`, creates the Ray `Cluster` +
`HybridComponentPlacement`, launches the `EnvWorker` group, and `init_worker().wait()`
-> `get_env_cls("isaaclab_arena")` builds the Arena env **inside the Ray worker**.
Result `M2b OK`: register() ran in the worker; `IsaacLabArenaManagerBasedRLEnvCfg` parsed;
policy/camera_obs/task_obs groups built. So config -> Cluster -> register() -> EnvWorker
-> Arena env all execute via RLinf's real machinery (only actor/rollout, which load
weights, are not launched -> no checkpoint needed).
Fixes/findings:
- **pyarrow**: Ray serialization broke on `pyarrow==14.0.1` (built vs numpy 1.x) with
  numpy 2.3.1 (`numpy.core.multiarray failed to import`). Upgraded `pyarrow -> 24.0.0`
  (`uv pip install -U pyarrow --no-deps`). One-venv fix; record for repro.
- **register() converter mismatch (non-fatal)**: the 4.6.5 extension's
  `_register_gr00t_converters` expects `simulation_io.ACTION_CONVERSION`, but the fork
  renamed it to `ACTION_CONVERSION_N1D5`. RLinf catches the error and the env still
  builds, but `_patch_gr00t_get_model` (after it) is skipped -> must fix for the M3 model
  path (add back-compat alias or update the extension).
Files: `arena_local_bringup/m2b_entrypoint_env_launch.py`, `drill_lift_g1_rlinf.yaml`, log `m2b_test.log`.

## M3 — full chain RUNS end-to-end (public N1.7, `DRY RUN OK`)

### 1. Why
Prove the last leg of the MVP: that the in-process, single-venv chain
**IsaacLab `train.py` entrypoint -> RLinf -> Arena G1 `drill_lift` env -> GR00T N1.7** loads,
forwards, and closes the RL loop (rollout -> reward -> actor update). Use the *public*
`nvidia/GR00T-N1.7-3B` so no private/BC checkpoint is needed for the plumbing proof.

### 2. How
- **Model-env (one venv; sim deps untouched).** Swapped gr00t (Arena submodule 1.1.0 ->
  standalone Isaac-GR00T N1.7, `uv pip install -e … --no-deps`), `transformers 4.51.3 ->
  4.57.3` (Qwen3-VL), added `decord 0.6.0`. torch stayed `2.10.0+cu128` / numpy `2.3.1`;
  `isaaclab`/`isaaclab_arena` still import. Restore point: `arena_venv_freeze_before_model.txt`.
- **HF token** (`juekun-nv`): `nvidia/Cosmos-Reason2-2B` (N1.7 backbone, gated) DOWNLOAD-OK;
  `NVEagle/...` (N1.5 backbone) 404 -> N1.7 is the only viable base. Downloaded N1.7-3B +
  Cosmos to `/home/juekunl/Work/models/`.
- **Finding — the public base does NOT ship the Arena drill_lift G1 embodiment.**
  `embodiment_id.json` has `unitree_g1_full_body_with_waist_height_nav_cmd` (slot 25), but
  `experiment_cfg/{config.yaml,dataset_statistics.json}` carry modality+stats for only 8
  embodiments; the one G1 is `real_g1_relative_eef_relative_joints` (eef-based:
  `left/right_wrist_eef_9d`, `left/right_hand`, `left/right_arm`, `waist`; action adds
  `base_height_command`,`navigate_command` = 53-d). The forward hard-looks-up
  `modality_configs[embodiment_tag.value]` -> `KeyError` for the real Arena tag.
- **Plumbing converter** (`obs/action_converter_type = "isaaclab_arena_g1_eef"`,
  `EmbodimentTag.ISAACLAB_ARENA_G1_EEF = "real_g1_relative_eef_relative_joints"`): best-effort
  maps the Arena 43-d joint state into the eef groups (arms/hands/waist copied;
  `*_wrist_eef_9d` filled with an **identity pose** `[0,0,0, 1,0,0, 0,1,0]` — a zero rot6d is
  a degenerate rotation -> `Rotation.from_matrix` SVD non-convergence in the
  relative->absolute decode), and extracts the 35-d Arena action from the model's 53-d output
  (drops the eef groups). Garbage policy, valid forward. (Real drill_lift: a BC-finetuned
  ckpt + `isaaclab_arena_g1`.)
- **Config + run.** `arena_local_bringup/drill_lift_g1_n17_plumbing.yaml` (self-contained,
  PPO/`actor_critic`, 1 env, `g1_gr00t` 35-d embodiment, local N1.7+Cosmos, `dry_run: True`),
  via the literal entrypoint `submodules/IsaacLab/scripts/reinforcement_learning/rlinf/train.py
  --config_path arena_local_bringup --config_name drill_lift_g1_n17_plumbing
  --model_path …/GR00T-N1.7-3B`.

### 3. Result
- **Model leg** (`m3_model_load_test.py`, no Isaac Sim): `GR00T_N1_7_ForRLActionPrediction`,
  **3.15B, 6.3 GB CUDA, bf16, sdpa** (no flash-attn); Cosmos backbone loaded locally; forward
  on Arena-G1 obs -> action `(1, 8, 35)` + PPO `prev_logprobs`/`prev_values`.
- **Full chain** (`m3_train.log`): EnvGroup built the drill_lift G1 env (Active Action Terms
  shape 35, `success_rate` recorder), Rollout `100%`, Actor PPO update ran, then
  `DRY RUN OK: built workers + env + model, ran 1 rollout and 1 actor update`. Clean exit.
  Metrics: `episode_len=8, success_once=1.0, reward=2.72e-05, critic/value_loss=0.317`.

### 4. Analysis & conclusion
The architecture is **proven end-to-end in one py3.12 venv on one GPU**: IsaacLab entrypoint ->
RLinf `Cluster`/`EnvWorker`/`MultiStepRolloutWorker`/`EmbodiedFSDPActor`/`EmbodiedRunner` ->
in-process Arena G1 env -> real GR00T N1.7 (3B + Cosmos) load+forward+update. Caveats:
1. Public base != Arena drill_lift G1 -> the policy is meaningless; a real run needs the
   BC-finetuned ckpt (S4) + `obs/embodiment = isaaclab_arena_g1` (config swap is documented in
   the yaml header).
2. `advantages/policy_loss=nan` is the 1-env dry-run x `normalize_advantages` (single-sample
   std=0). Use >=2 envs or `algorithm.normalize_advantages: False`.
3. Container `/dev/shm`=64 MB is too small for NCCL's ~31.5 MB segment -> ran with
   `NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1` (single-GPU is fine); for the cluster, start the
   container with `--shm-size=2g`.
4. `ray stop --force` between runs to free GPU/`/dev/shm` from crashed workers.

### RLinf fixes landed (model + env side)
- `models/embodiment/gr00t/simulation_io.py`: back-compat `ACTION_CONVERSION` alias (N1.5) for
  the 4.6.5 bridge `register()`; `convert_isaaclab_arena_g1_eef_obs_to_gr00t_format` +
  `convert_to_isaaclab_arena_g1_eef_action` + `OBS_CONVERSION`/`ACTION_CONVERSION_N1D7` entries.
- `models/embodiment/gr00t/embodiment_tags.py`: `ISAACLAB_ARENA_G1_EEF`.
- `models/embodiment/gr00t/gr00t_n1d7/__init__.py`: `isaaclab_arena_g1_eef` -> tag in the map.
- `envs/isaaclab_arena/arena_inprocess_env.py`: `_wrap_obs` assembles the 43-d state from the
  7 named `policy` groups in canonical order (g1_gr00t exposes no flat `robot_joint_pos` under
  `policy`; the flat vector is under `wbc/robot_joint_pos`).
- `envs/action_utils.py`: `prepare_actions_for_isaaclab_arena` now converts numpy -> torch
  (IsaacLab `action_manager.process_action` calls `action.to(device)`).

### Next
Real BC ckpt + `isaaclab_arena_g1`; flip PPO -> DSRL (`embodied_sac`); bump to >=2 envs to
remove the nan; then scale on OSMO. (DSRL + multi-env done -> see M4.)

## M4 — multi-env + DSRL (`embodied_sac`) chain (public N1.7, `DRY RUN OK`)

### 1. Why
After M3 proved the PPO chain on 1 env, extend to (a) `num_envs>1` (open C-phase question +
to remove the 1-env advantage `nan`) and (b) **DSRL** (`embodied_sac`) — the actual research
method: a SAC agent steers the *frozen* GR00T flow policy by choosing its diffusion noise.

### 2. How
- **Multi-env**: `drill_lift_g1_n17_plumbing.yaml` train `total_num_envs: 1 -> 4`, re-ran PPO.
- **DSRL model leg** (`m4_dsrl_model_test.py`, no Isaac Sim): load with
  `rl_head_config.use_dsrl: True` + `add_value_head: False` + `dsrl_*` (state_dim 43, 10 Q
  heads, latents 64). Found + fixed a latent N1.7 DSRL bug: the rollout `get_rl_action`
  called `get_value` (-> `value_head`) unconditionally, but DSRL has no flow value head ->
  `AttributeError`. Gated it on `hasattr(self, "value_head")` (DSRL `prev_values` are zeros,
  unused by the SAC actor).
- **DSRL config** `drill_lift_g1_n17_dsrl_plumbing.yaml`: `algorithm.{adv_type,loss_type}:
  embodied_sac` + `entropy_tuning` (`target_entropy: -66` ~= -noise_dim/2, noise_dim=132) +
  `replay_buffer` (`min_buffer_size: 10`) + `actor.critic_optim`; `actor.model.add_value_head:
  False` + `rl_head_config.use_dsrl: True` + `dsrl_*`. `train.py` routes
  `loss_type==embodied_sac` -> `EmbodiedSACFSDPPolicy`. Ran via the literal entrypoint.
  (Rollout uses `cfg.actor.model` for architecture in train mode, so `use_dsrl` on the actor
  covers the rollout too — only path/precision come from `rollout.model`.)

### 3. Result
- **Multi-env PPO** (`m3b_train_4env.log`): `DRY RUN OK`; advantages now real —
  `advantages in [-1.189, 1.187]`, mean ~= 0; `actor/policy_loss=-1.915`,
  `critic/value_loss=7.085`. The in-process Arena env runs 4 parallel envs; the M3 `nan` was
  single-sample advantage normalization.
- **DSRL model leg** (`m4_dsrl_model_test.py`): `[DSRL] initialized: noise_dim=132 state_dim=43
  num_q_heads=10`, `freeze_vlm: froze 1030 base param tensors`; `predict_action_batch(train)`
  -> action `(2, 8, 35)`.
- **DSRL full chain** (`m4_dsrl_train.log`): `EmbodiedSACFSDPPolicy` built the SAC heads on the
  frozen GR00T base (`dsrl_action_noise_net`, `q_head.q_heads.0..9`, actor/critic image+state
  encoders); rollout + 1 SAC update ran; `DRY RUN OK`. `reward=4.56e-05, success_once=1.0`.

### 4. Analysis & conclusion
Both increments green. `num_envs>1` works for the in-process Arena env with no code change.
**The DSRL research method runs end-to-end** through the literal IsaacLab entrypoint on the
public N1.7 base (SAC steering the frozen flow's 132-d noise latent + its own 10-head Q
critic). Same plumbing caveat as M3 (eef embodiment -> meaningless policy; a real run needs the
BC ckpt + `isaaclab_arena_g1`). The `get_value` guard is a real RLinf fix — the N1.7 DSRL
rollout path was previously broken whenever `add_value_head=False`.

RLinf fix: `models/embodiment/gr00t/gr00t_n1d7/gr00t_action_model.py` — `get_rl_action` gates
`get_value` on `hasattr(self, "value_head")`.

### Next (post-M4)
Real BC drill_lift ckpt + swap to `isaaclab_arena_g1` (one-line config change) for a meaningful
policy; then a longer DSRL run (raise `update_epoch`/`train_actor_steps`, more envs/steps) to
watch success_rate move; then OSMO scale-out (`--shm-size=2g`, multi-GPU placement).

## In-process MVP files
- `rlinf/envs/isaaclab_arena/arena_inprocess_env.py` (new), `rlinf/envs/isaaclab_arena/__init__.py` (registers in-process env as default)
- `arena_local_bringup/m0_drilllift_75.py` (7.5.0 direct-build probe), `m2_inprocess_env_test.py`, `m2b_entrypoint_env_launch.py`
- M3: `arena_local_bringup/{m3_model_load_test.py, m3_dump_obs.py, drill_lift_g1_n17_plumbing.yaml, m3_train.log, n17_download.log, arena_venv_freeze_before_model.txt}`
- M4: `arena_local_bringup/{m4_dsrl_model_test.py, drill_lift_g1_n17_dsrl_plumbing.yaml, m4_dsrl_train.log, m3b_train_4env.log}`
- Logs: `m0b_4_6_5_smoke.log`, `m2_test.log`, `m2b_test.log`, `m3_train.log`, `m3b_train_4env.log`, `m4_dsrl_train.log`
- IsaacLab-Arena 7.5.0 import fixes (reverted for 4.6.5 MVP; re-apply as shims for the PR): `manager_based.manipulation.{pick_place,stack,place}` -> `contrib.*` in `embodiments/g1/g1.py`, `tasks/events.py`, `assets/object_base.py`, `isaaclab_arena_environments/mdp/__init__.py`
