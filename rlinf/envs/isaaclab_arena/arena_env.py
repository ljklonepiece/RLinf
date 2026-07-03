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

"""RLinf env wrapper for the IsaacLab-Arena G1 locomanipulation suite (Option B).

This hosts an Arena env (e.g. ``LMDrillLiftRlD1``) for RLinf. It is deliberately model-
and algorithm-agnostic: the Arena sim is built with the ``g1`` embodiment (35-dim
joint-space action) and emits RLinf's canonical observation dict
(``main_images`` / ``states`` / ``task_descriptions``). Mapping that canonical obs into a
specific policy's input layout is the job of the model-side ``obs_converter`` (e.g.
``obs_converter_type: isaaclab_arena_g1``), not this env.

**Why a proxy, not an in-process subprocess env.** RLinf runs on CPython 3.11 while
IsaacLab-Arena (``isaaclab==4.6.5``) requires CPython 3.12, and Ray refuses to join a
3.12 worker to a 3.11 driver (Python *minor* parity, proven by the S0.4 Ray probe). So
``IsaaclabBaseEnv``'s ``SubProcIsaacLabEnv`` (which inherits the parent's py3.11 ABI via
``spawn``) cannot load the cp312 Arena extensions. Instead the Arena sim runs in its own
py3.12 process (``arena_server.py`` in the Arena ``.venv``), and this class drives it over
a local socket exchanging numpy/CPU payloads (see ``arena_client.py`` / ``protocol.py``).
This mirrors the posttrain DSRL setup (separate TCP Arena sim servers).

All the generic RLinf env behaviour (metrics, ``step`` / ``chunk_step`` / reward shaping /
elapsed-step tracking) is reused unchanged from ``IsaaclabBaseEnv``; only env
construction (``_init_isaaclab_env``) and obs conversion (``_wrap_obs``) are overridden,
because the heavy lifting (state assembly, camera selection) now happens server-side in
``g1_layout.wrap_obs``.
"""

from __future__ import annotations

import torch

from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

from .arena_client import ArenaSocketEnv


class IsaaclabArenaG1Env(IsaaclabBaseEnv):
    """IsaacLab-Arena G1 locomanip env (drill lift, box lift, ...) for RLinf."""

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )

    def _init_isaaclab_env(self):
        """Build the socket-client env instead of an in-process subprocess env.

        Connection settings live under ``cfg.init_params.server``:
            host / port              -- connect to an already-running Arena server.
            auto_spawn / spawn_cmd / ready_file
                                     -- launch a local server (e.g. via the docker
                                        helper) and connect once its ready-file
                                        publishes the bound port.
        ``proxy_device`` (default ``cpu``) is where obs/reward tensors live on the RLinf
        side; the sim itself runs on ``init_params.device`` inside the server.
        ``reward_mode`` (default ``sparse``) selects the RL reward: ``sparse`` uses the
        env's ``success`` termination term (1/0), ``shaped`` uses Arena's dense reward.
        """
        ip = self.cfg.init_params
        server = ip.get("server", {}) or {}

        spawn_cmd = server.get("spawn_cmd", None)
        if spawn_cmd is not None:
            spawn_cmd = list(spawn_cmd)

        self.env = ArenaSocketEnv(
            env_name=self.isaaclab_env_id,
            num_envs=self.num_envs,
            sim_device=ip.get("device", "cuda:0"),
            task_description=ip.task_description,
            reward_mode=ip.get("reward_mode", "sparse"),
            host=server.get("host", "127.0.0.1"),
            port=int(server.get("port", 5557)),
            proxy_device=ip.get("proxy_device", "cpu"),
            auto_spawn=bool(server.get("auto_spawn", False)),
            spawn_cmd=spawn_cmd,
            ready_file=server.get("ready_file", None),
            connect_timeout=float(server.get("connect_timeout", 600.0)),
        )
        self.env.reset(seed=self.seed)

    def _wrap_obs(self, obs):
        """Convert the server's canonical numpy obs dict to torch on ``self.device``.

        The server (``g1_layout.wrap_obs``) already assembled the flat 43-d G1 state in
        the correct group order and selected the single ego camera, so here we only move
        numpy -> torch. ``task_descriptions`` stays a list of strings.
        """
        env_obs = {
            "main_images": torch.from_numpy(obs["main_images"]).to(self.device),
            "states": torch.from_numpy(obs["states"]).to(self.device),
            "task_descriptions": list(obs["task_descriptions"]),
        }
        return env_obs
