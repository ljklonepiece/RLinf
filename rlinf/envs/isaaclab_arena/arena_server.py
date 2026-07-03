# Copyright 2025 The RLinf Authors.
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

"""Out-of-process IsaacLab-Arena env server (CPython 3.12).

RLinf runs on CPython 3.11; IsaacLab-Arena (``isaaclab==4.6.5``, cp312) requires
CPython 3.12. Ray refuses to join workers of a different Python minor to one cluster,
so the Arena env cannot live inside an RLinf Ray worker. Instead this script hosts the
Arena env in its **own** py3.12 process (the Arena ``.venv``), and the RLinf py3.11 env
worker drives it over a local socket exchanging numpy payloads (see ``protocol.py``).
This mirrors the posttrain DSRL setup (separate TCP Arena sim servers).

Run it with the Arena venv interpreter, as a *file* (so sibling modules import without
pulling in the ``rlinf`` package, which would fail under the Arena venv):

    /path/IsaacLab-Arena/.venv/bin/python -u \
        /path/RLinf/rlinf/envs/isaaclab_arena/arena_server.py \
        --env-name LMDrillLiftRlD1 --num-envs 1 --port 5557 \
        --ready-file /tmp/arena_server.ready

Headless render parity on this workstation requires running inside the
``gear-n2-eval`` container (see ``arena_local_bringup/run_arena_server.sh``); the bare
host headless path crashes in ``reset()``.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback

import numpy as np

# Run-as-a-file sibling imports: make this dir importable WITHOUT importing the
# ``rlinf`` package (its top-level deps are py3.11/torch-2.6 and absent here).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import protocol  # noqa: E402  (sibling module)
from bootstrap import launch_isaac_app  # noqa: E402  (sibling module)
from cli_defaults import isaaclab_arena_cli_defaults  # noqa: E402  (sibling module)
from g1_layout import to_numpy, wrap_obs  # noqa: E402  (sibling module)

# The IsaacLab termination term that means "task solved". For the G1 lift suite it is
# ``object_lifted & is_grasped`` (see IsaacLab-Arena g1_locomanip_lift_task.py). We read
# it explicitly rather than using ``terminated`` because ``terminated`` also fires on
# failure terms like ``object_dropping``; mirrors gr00t's IsaacLabArenaEnvWrapper.
SUCCESS_TERM_NAME = "success"

# Robust debug trace to a FILE (not stdout): Omniverse Kit can capture/redirect stdout,
# so a hard crash inside env.step (segfault, no Python traceback) may otherwise hide which
# step/action triggered it. A plain file write survives. Off by default cost is negligible.
_DBG_FILE = os.environ.get("ARENA_DBG_FILE", "/tmp/arena_dbg.log")


def _dbg(msg: str) -> None:
    try:
        with open(_DBG_FILE, "a") as _f:
            _f.write(msg + "\n")
            _f.flush()
    except Exception:  # noqa: BLE001
        pass


def _build_env(env_name: str, num_envs: int, device: str):
    """Bootstrap Kit and build the Arena gym env. Returns (env, sim_app)."""
    sim_app = launch_isaac_app(headless=True, enable_cameras=True)

    defaults = isaaclab_arena_cli_defaults(env_name)
    defaults["num_envs"] = num_envs
    defaults["device"] = device
    args = argparse.Namespace(**defaults)

    from isaaclab_arena_environments.cli import get_arena_builder_from_cli

    builder = get_arena_builder_from_cli(args)

    import gymnasium as gym
    from isaaclab.managers.recorder_manager import (
        DatasetExportMode as _DatasetExportMode,
    )

    reg_name, env_cfg = builder.build_registered()
    env_cfg.recorders.dataset_export_mode = _DatasetExportMode.EXPORT_NONE
    env = gym.make(reg_name, cfg=env_cfg).unwrapped
    return env, sim_app


def _action_dim(env) -> int:
    sp = getattr(env, "action_space", None)
    if sp is not None and getattr(sp, "shape", None):
        return int(sp.shape[-1])
    am = getattr(env, "action_manager", None)
    if am is not None and hasattr(am, "total_action_dim"):
        return int(am.total_action_dim)
    raise RuntimeError("Could not determine Arena action dim from env.")


def _termination_manager(env):
    """Return the IsaacLab ``TerminationManager`` of the (possibly wrapped) env, or None."""
    base = getattr(env, "unwrapped", env)
    return getattr(base, "termination_manager", None)


def _has_success_term(env) -> bool:
    """Whether the env exposes a termination term named ``success`` (sparse-reward signal)."""
    term_mgr = _termination_manager(env)
    if term_mgr is None:
        return False
    try:
        return SUCCESS_TERM_NAME in term_mgr.active_terms
    except Exception:  # noqa: BLE001
        return False


def _extract_success(env, num_envs: int) -> np.ndarray:
    """Read the per-env ``success`` termination term as a numpy bool array (num_envs,).

    This is the *true* sparse success signal: the IsaacLab termination term literally
    named ``success`` (object lifted + grasped), distinct from ``terminated`` (which also
    fires on failure terms) and from ``reward > 0``. ManagerBasedRLEnv may internally
    reset terminated envs during ``step``, but the TerminationManager retains this step's
    per-term values, so reading right after ``step`` is correct (matches gr00t). Falls
    back to all-False if the term is absent.
    """
    term_mgr = _termination_manager(env)
    if term_mgr is not None:
        try:
            if SUCCESS_TERM_NAME in term_mgr.active_terms:
                val = term_mgr.get_term(SUCCESS_TERM_NAME)
                return to_numpy(val).reshape(num_envs).astype(bool)
        except Exception:  # noqa: BLE001  (best-effort; fall through to all-False)
            pass
    return np.zeros((num_envs,), dtype=bool)


def _sanitize_info(info) -> dict:
    """Keep only picklable, lightweight entries from the env extras dict."""
    out: dict = {}
    if not isinstance(info, dict):
        return out
    for k, v in info.items():
        try:
            if hasattr(v, "detach") or hasattr(v, "cpu"):
                out[k] = to_numpy(v)
            elif isinstance(v, (int, float, bool, str)):
                out[k] = v
        except Exception:  # noqa: BLE001  (best-effort; drop anything odd)
            continue
    return out


class ArenaEnvServer:
    def __init__(self, env_name: str, num_envs: int, device: str, task_description: str):
        self.env_name = env_name
        self.num_envs = num_envs
        self.device = device
        self.task_description = task_description
        print(f"[arena_server] building env={env_name} num_envs={num_envs} device={device}", flush=True)
        self.env, self.sim_app = _build_env(env_name, num_envs, device)
        self.action_dim = _action_dim(self.env)
        self.has_success = _has_success_term(self.env)
        print(f"[arena_server] env built. action_dim={self.action_dim} "
              f"has_success={self.has_success}", flush=True)
        if not self.has_success:
            # Sparse reward depends on this term; warn loudly so a misconfigured task is
            # caught at boot rather than producing an all-zero (no-signal) reward.
            print(
                f"[arena_server] WARNING: no '{SUCCESS_TERM_NAME}' termination term found; "
                "sparse reward would be all-zeros. Use reward_mode='shaped' for this env.",
                flush=True,
            )

    # --- command handlers -------------------------------------------------
    def handle_spec(self, _msg) -> dict:
        return {
            protocol.KEY_OK: True,
            protocol.KEY_SPEC: {
                "env_name": self.env_name,
                "num_envs": self.num_envs,
                "action_dim": self.action_dim,
                "has_success": self.has_success,
            },
        }

    def handle_reset(self, _msg) -> dict:
        obs, info = self.env.reset()
        wrapped = wrap_obs(obs, self.task_description, self.num_envs)
        try:
            _s = np.asarray(wrapped["states"], dtype=np.float64)
            _img = np.asarray(wrapped["main_images"])
            _dbg(
                f"reset state shape={_s.shape} min={_s.min():.3f} max={_s.max():.3f} "
                f"nan={int(np.isnan(_s).sum())} | img shape={_img.shape} "
                f"dtype={_img.dtype} min={int(_img.min())} max={int(_img.max())}"
            )
        except Exception as _e:  # noqa: BLE001
            _dbg(f"reset obs stat error: {_e!r}")
        return {
            protocol.KEY_OK: True,
            protocol.KEY_OBS: wrapped,
            protocol.KEY_INFO: _sanitize_info(info),
        }

    def handle_step(self, msg) -> dict:
        import torch

        action = msg["action"]  # numpy (num_envs, action_dim)
        # Robust per-step trace to a FILE so a hard sim crash inside env.step (no Python
        # traceback, stdout possibly swallowed by Kit) can be pinned to the exact action.
        self._step_count = getattr(self, "_step_count", 0) + 1
        try:
            _a = np.asarray(action, dtype=np.float64)
            _dbg(
                f"step#{self._step_count} action shape={_a.shape} "
                f"min={_a.min():.4f} max={_a.max():.4f} mean={_a.mean():.4f} "
                f"nan={int(np.isnan(_a).sum())} inf={int(np.isinf(_a).sum())} "
                f"head={np.round(_a.reshape(-1)[:12], 3).tolist()}"
            )
        except Exception as _e:  # noqa: BLE001
            _dbg(f"step#{self._step_count} action stat error: {_e!r}")
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        _dbg(f"step#{self._step_count} -> entering env.step")
        obs, reward, terminated, truncated, info = self.env.step(action_t)
        _dbg(f"step#{self._step_count} <- env.step returned")
        # Emit BOTH the shaped reward and the explicit success term; the RLinf-side
        # proxy picks which to use as the RL reward (reward_mode). terminated/truncated
        # are always the true signals so the learner's done/bootstrap stays correct.
        success = _extract_success(self.env, self.num_envs)
        return {
            protocol.KEY_OK: True,
            protocol.KEY_OBS: wrap_obs(obs, self.task_description, self.num_envs),
            protocol.KEY_REWARD: to_numpy(reward).reshape(self.num_envs),
            protocol.KEY_TERMINATED: to_numpy(terminated).reshape(self.num_envs).astype(bool),
            protocol.KEY_TRUNCATED: to_numpy(truncated).reshape(self.num_envs).astype(bool),
            protocol.KEY_SUCCESS: success,
            protocol.KEY_INFO: _sanitize_info(info),
        }

    def dispatch(self, msg) -> dict:
        cmd = msg.get("cmd")
        try:
            if cmd == protocol.CMD_PING:
                return {protocol.KEY_OK: True}
            if cmd == protocol.CMD_SPEC:
                return self.handle_spec(msg)
            if cmd == protocol.CMD_RESET:
                return self.handle_reset(msg)
            if cmd == protocol.CMD_STEP:
                return self.handle_step(msg)
            return {protocol.KEY_OK: False, protocol.KEY_ERROR: f"unknown cmd {cmd!r}"}
        except Exception:  # noqa: BLE001  (surface failures to the driver, don't hang)
            return {protocol.KEY_OK: False, protocol.KEY_ERROR: traceback.format_exc()}

    def close(self):
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.sim_app.close()
        except Exception:  # noqa: BLE001
            pass


def serve(args) -> int:
    server = ArenaEnvServer(
        env_name=args.env_name,
        num_envs=args.num_envs,
        device=args.device,
        task_description=args.task_description,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    bound_port = sock.getsockname()[1]
    print(f"[arena_server] listening on {args.host}:{bound_port}", flush=True)

    if args.ready_file:
        # Write the bound port last so the parent can poll for readiness atomically.
        tmp = f"{args.ready_file}.tmp"
        with open(tmp, "w") as f:
            f.write(str(bound_port))
        os.replace(tmp, args.ready_file)
        print(f"[arena_server] wrote ready-file {args.ready_file} ({bound_port})", flush=True)

    try:
        while True:
            conn, addr = sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[arena_server] client connected from {addr}", flush=True)
            try:
                while True:
                    msg = protocol.recv_msg(conn)
                    _dbg(f"recv cmd={msg.get('cmd')!r}")
                    if msg.get("cmd") == protocol.CMD_CLOSE:
                        protocol.send_msg(conn, {protocol.KEY_OK: True})
                        print("[arena_server] received CLOSE; shutting down", flush=True)
                        return 0
                    protocol.send_msg(conn, server.dispatch(msg))
            except (ConnectionError, EOFError) as e:
                print(f"[arena_server] client disconnected: {e!r}", flush=True)
            finally:
                conn.close()
    finally:
        server.close()
        sock.close()


def main() -> int:
    p = argparse.ArgumentParser(description="IsaacLab-Arena env socket server (py3.12)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 = ephemeral; see --ready-file")
    p.add_argument("--ready-file", default="", help="path to write the bound port when ready")
    p.add_argument("--env-name", default="LMDrillLiftRlD1")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--task-description",
        default="Pick up the power drill from the table and lift it.",
    )
    args = p.parse_args()
    return serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
