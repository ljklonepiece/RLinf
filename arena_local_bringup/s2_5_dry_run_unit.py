#!/usr/bin/env python3
"""S2.5 unit check: the runner --dry-run gate (control flow, no GPU/model/env).

The dry-run is the pre-submit gate: build everything, run exactly ONE rollout + ONE
actor update, skip val/eval/checkpoint, print 'DRY RUN OK', exit 0. Running it for real
needs the full stack (workers/model/env/GPU) and is exercised at S3. Here we verify the
*control flow* of ``EmbodiedRunner.run()`` in isolation by building the runner via
``__new__`` and stubbing the heavy collaborators (worker groups -> fake handles, timer,
loggers). We assert:

  * dry_run=True  -> exactly ONE step; ``_maybe_eval_and_checkpoint`` NOT called;
                     ``_save_checkpoint`` NOT called; "DRY RUN OK" logged; clean finish.
  * dry_run=False -> (with max_steps=1) ONE step AND ``_maybe_eval_and_checkpoint`` IS
                     called -> proves the gate only skips eval/ckpt under dry_run.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s2_5_dry_run_unit.py
"""

from __future__ import annotations

import contextlib
import sys

from omegaconf import OmegaConf

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.runners.embodied_runner import EmbodiedRunner  # noqa: E402

_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not cond:
        _FAILURES.append(name)


class _FakeHandle:
    def wait(self, *a, **k):
        return {}

    def consume_durations(self, *a, **k):
        return ({}, {})


class _FakeGroup:
    """Any method returns a fake Handle; covers actor/rollout/env collaborators."""

    def __getattr__(self, _name):
        def _method(*a, **k):
            return _FakeHandle()
        return _method


class _FakeTimer:
    def __call__(self, _name):
        return contextlib.nullcontext()


class _SpyLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *a, **k):
        self.messages.append(str(msg))


def _build_runner(dry_run: bool, max_steps: int) -> tuple[EmbodiedRunner, dict]:
    r = EmbodiedRunner.__new__(EmbodiedRunner)
    r.cfg = OmegaConf.create(
        {"runner": {"dry_run": dry_run, "use_training_pipeline": False}}
    )
    r.global_step = 0
    r.max_steps = max_steps
    r.weight_sync_interval = 1
    r.overlap_env_bootstrap = False
    r.actor = _FakeGroup()
    r.rollout = _FakeGroup()
    r.env = _FakeGroup()
    r.reward = None
    r.env_channel = r.rollout_channel = r.reward_channel = r.actor_channel = None
    r.timer = _FakeTimer()
    r.logger = _SpyLogger()

    counters = {"eval_calls": 0, "save_calls": 0, "log_step_calls": 0, "finish_calls": 0}

    def _maybe_eval(step):
        counters["eval_calls"] += 1
        return {}

    def _save():
        counters["save_calls"] += 1

    def _log_step(**kw):
        counters["log_step_calls"] += 1

    def _finish():
        counters["finish_calls"] += 1

    r._maybe_eval_and_checkpoint = _maybe_eval
    r._save_checkpoint = _save
    r._log_step_metrics = _log_step
    r._finish_run = _finish
    r._should_profile_step = lambda _s: False
    r.update_rollout_weights = lambda *a, **k: None
    return r, counters


def test_dry_run_true() -> None:
    r, c = _build_runner(dry_run=True, max_steps=100)
    r.run()
    check("dry.one_step", r.global_step == 1, f"global_step={r.global_step}")
    check("dry.eval_skipped", c["eval_calls"] == 0, f"eval_calls={c['eval_calls']}")
    check("dry.checkpoint_skipped", c["save_calls"] == 0)
    check("dry.finish_called", c["finish_calls"] == 1)
    check("dry.prints_DRY_RUN_OK",
          any("DRY RUN OK" in m for m in r.logger.messages),
          f"logs={r.logger.messages}")


def test_normal_run_evals() -> None:
    # max_steps=1 so it also runs a single step, but WITHOUT dry_run -> eval/ckpt path runs
    r, c = _build_runner(dry_run=False, max_steps=1)
    r.run()
    check("normal.one_step", r.global_step == 1, f"global_step={r.global_step}")
    check("normal.eval_called", c["eval_calls"] == 1, f"eval_calls={c['eval_calls']}")
    check("normal.no_DRY_RUN_OK",
          not any("DRY RUN OK" in m for m in r.logger.messages))


def main() -> int:
    test_dry_run_true()
    test_normal_run_evals()
    print()
    if _FAILURES:
        print(f"S2.5 DRY-RUN FAILED ({len(_FAILURES)}): {_FAILURES}")
        return 1
    print("S2.5 DRY-RUN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
