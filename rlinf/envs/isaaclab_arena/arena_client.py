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

"""Socket-client env that drives an out-of-process IsaacLab-Arena server.

This is the RLinf-side (CPython 3.11) half of the Option-B integration: it speaks
the ``protocol`` wire to ``arena_server.py`` (CPython 3.12) and presents the **same
interface as ``SubProcIsaacLabEnv``** (``reset`` / ``step`` / ``device`` / ``close``),
so ``IsaaclabArenaG1Env`` can use it as a drop-in for the in-process subprocess env.

Boundary contract (must stay numpy/CPU): observations arrive as numpy arrays and are
handed up as-is to the env wrapper's ``_wrap_obs`` (which converts to torch); reward /
terminated / truncated are converted to torch tensors on ``proxy_device`` (default CPU,
since the sim runs remotely and obs cross Ray worker boundaries as plain arrays).

Two lifecycles:
- **connect**: attach to an already-running server at ``host:port`` (cluster /
  posttrain-style external sim servers).
- **spawn**: launch a local server via ``spawn_cmd`` (e.g. the docker helper), wait for
  its ready-file to publish the bound port, then connect; terminated on ``close``.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Optional

import numpy as np
import torch

try:  # package import (normal RLinf use)
    from . import protocol
except ImportError:  # run-as-file / standalone import (unit checks)
    import protocol  # type: ignore


class ArenaSocketEnv:
    """Drop-in for ``SubProcIsaacLabEnv`` backed by a remote py3.12 Arena server."""

    def __init__(
        self,
        *,
        env_name: str,
        num_envs: int,
        sim_device: str = "cuda:0",
        task_description: str = "",
        reward_mode: str = "sparse",
        host: str = "127.0.0.1",
        port: int = 5557,
        proxy_device: str = "cpu",
        auto_spawn: bool = False,
        spawn_cmd: Optional[list[str]] = None,
        ready_file: Optional[str] = None,
        connect_timeout: float = 600.0,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.sim_device = sim_device
        self.task_description = task_description
        if reward_mode not in ("sparse", "shaped"):
            raise ValueError(
                f"reward_mode must be 'sparse' or 'shaped', got {reward_mode!r}."
            )
        self.reward_mode = reward_mode
        self._proxy_device = torch.device(proxy_device)
        self._proc: Optional[subprocess.Popen] = None

        if auto_spawn:
            if not spawn_cmd or not ready_file:
                raise ValueError(
                    "auto_spawn requires both spawn_cmd and ready_file."
                )
            host, port = self._spawn_server(spawn_cmd, ready_file, connect_timeout)

        self._sock = self._connect(host, port, connect_timeout)

        spec = self._rpc({"cmd": protocol.CMD_SPEC})[protocol.KEY_SPEC]
        self.action_dim = int(spec["action_dim"])
        self.has_success = bool(spec.get("has_success", False))
        server_num_envs = int(spec["num_envs"])
        if server_num_envs != num_envs:
            raise RuntimeError(
                f"Arena server num_envs={server_num_envs} != requested {num_envs}. "
                "Spawn the server with matching ARENA_NUM_ENVS."
            )
        # Fail fast: sparse reward needs the env's 'success' termination term. Without it
        # the reward would be silently all-zeros (no learning signal).
        if self.reward_mode == "sparse" and not self.has_success:
            raise RuntimeError(
                f"reward_mode='sparse' but Arena env {env_name!r} reports no 'success' "
                "termination term (spec.has_success=False). Use reward_mode='shaped', or "
                "use a task that defines a 'success' termination term."
            )

    # --- lifecycle --------------------------------------------------------
    def _spawn_server(
        self, spawn_cmd: list[str], ready_file: str, timeout: float
    ) -> tuple[str, int]:
        if os.path.exists(ready_file):
            os.remove(ready_file)
        self._proc = subprocess.Popen(spawn_cmd)  # noqa: S603
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"Arena server process exited early (code {self._proc.returncode}) "
                    f"before writing ready-file {ready_file}."
                )
            if os.path.exists(ready_file):
                with open(ready_file) as f:
                    port = int(f.read().strip())
                return "127.0.0.1", port
            time.sleep(2.0)
        raise TimeoutError(f"Arena server ready-file {ready_file} never appeared.")

    def _connect(self, host: str, port: int, timeout: float) -> socket.socket:
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection((host, port), timeout=120)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return sock
            except OSError as e:  # server not up yet
                last_err = e
                time.sleep(2.0)
        raise TimeoutError(
            f"Could not connect to Arena server {host}:{port} within {timeout}s "
            f"(last error: {last_err!r})."
        )

    # --- rpc --------------------------------------------------------------
    def _rpc(self, msg: dict) -> dict:
        protocol.send_msg(self._sock, msg)
        resp = protocol.recv_msg(self._sock)
        if not resp.get(protocol.KEY_OK):
            raise RuntimeError(
                f"Arena server error for cmd={msg.get('cmd')!r}:\n"
                f"{resp.get(protocol.KEY_ERROR)}"
            )
        return resp

    # --- SubProcIsaacLabEnv-compatible API --------------------------------
    def reset(self, seed=None, env_ids=None):
        if env_ids is not None:
            # The Arena server resets all envs at once; partial env_ids reset is only
            # requested by auto_reset, which is disabled for this env (auto_reset=False).
            raise NotImplementedError(
                "ArenaSocketEnv does not support partial env_ids reset; "
                "run with auto_reset=False."
            )
        resp = self._rpc({"cmd": protocol.CMD_RESET})
        return resp[protocol.KEY_OBS], resp.get(protocol.KEY_INFO, {})

    def step(self, action: torch.Tensor):
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.asarray(action)
        action_np = np.ascontiguousarray(action_np, dtype=np.float32)

        resp = self._rpc({"cmd": protocol.CMD_STEP, "action": action_np})
        obs = resp[protocol.KEY_OBS]
        shaped_reward = torch.as_tensor(
            resp[protocol.KEY_REWARD], dtype=torch.float32, device=self._proxy_device
        )
        terminated = torch.as_tensor(
            resp[protocol.KEY_TERMINATED], dtype=torch.bool, device=self._proxy_device
        )
        truncated = torch.as_tensor(
            resp[protocol.KEY_TRUNCATED], dtype=torch.bool, device=self._proxy_device
        )
        success_np = resp.get(protocol.KEY_SUCCESS)
        if success_np is not None:
            success = torch.as_tensor(
                success_np, dtype=torch.bool, device=self._proxy_device
            )
        else:  # older server without the success channel
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self._proxy_device)

        # Reward formulation. terminated/truncated are always the true signals (kept
        # separate for the learner's done/bootstrap); only the reward channel changes.
        #   sparse  -> reward = success (1.0 on the success step, 0.0 else). This is the
        #             generic, task-agnostic signal and matches gr00t's IQL reward_mode.
        #             It also makes RLinf's inherited success metric (step_reward > 0)
        #             coincide with true success.
        #   shaped  -> reward = Arena's dense per-step reward_buf (approach/reach/lift/...).
        if self.reward_mode == "sparse":
            reward = success.float()
        else:
            reward = shaped_reward

        infos = resp.get(protocol.KEY_INFO, {}) or {}
        infos["success"] = success  # always surface the true success signal for logging
        return obs, reward, terminated, truncated, infos

    def device(self) -> torch.device:
        return self._proxy_device

    def close(self):
        try:
            protocol.send_msg(self._sock, {"cmd": protocol.CMD_CLOSE})
            protocol.recv_msg(self._sock)
        except (OSError, ConnectionError, EOFError):
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
