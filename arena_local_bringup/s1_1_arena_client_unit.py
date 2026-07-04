#!/usr/bin/env python3
"""S1.1 unit check: ArenaSocketEnv transport against a fake in-process server.

Isolates the *new* S1.1 code (``ArenaSocketEnv``, the py3.11 socket-client env) from
the real Docker/Isaac server (already proven end-to-end in B1). A tiny thread-based
server speaks the same ``protocol`` wire and returns canonical numpy obs, so this runs
fast and deterministically with no GPU/Isaac. It verifies:

  * connect + SPEC handshake (action_dim / num_envs)
  * reset() returns the canonical numpy obs dict
  * step() converts the torch action -> float32 numpy on the wire, and returns
    reward/terminated/truncated as torch tensors on the proxy device (CPU)
  * the server actually receives a (num_envs, action_dim) float32 action
  * clean close()

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s1_1_arena_client_unit.py
"""

from __future__ import annotations

import socket
import sys
import threading

import numpy as np
import torch

RLINF = "/home/juekunl/Work/RLinf"
sys.path.insert(0, RLINF)

from rlinf.envs.isaaclab_arena import protocol  # noqa: E402
from rlinf.envs.isaaclab_arena.arena_client import ArenaSocketEnv  # noqa: E402

NUM_ENVS = 1
ACTION_DIM = 35
STATE_DIM = 43
IMG_HW = (480, 640, 3)
TASK = "Pick up the power drill from the table and lift it."

# Captured by the fake server so the client side can assert on it.
_received = {}


def _canonical_obs() -> dict:
    return {
        "main_images": np.zeros((NUM_ENVS, *IMG_HW), dtype=np.uint8),
        "states": np.arange(NUM_ENVS * STATE_DIM, dtype=np.float32).reshape(
            NUM_ENVS, STATE_DIM
        ),
        "task_descriptions": [TASK] * NUM_ENVS,
    }


def _fake_server(sock: socket.socket) -> None:
    conn, _ = sock.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        while True:
            msg = protocol.recv_msg(conn)
            cmd = msg.get("cmd")
            if cmd == protocol.CMD_SPEC:
                protocol.send_msg(
                    conn,
                    {
                        protocol.KEY_OK: True,
                        protocol.KEY_SPEC: {
                            "env_name": "FakeDrillLift",
                            "num_envs": NUM_ENVS,
                            "action_dim": ACTION_DIM,
                            "has_success": True,
                        },
                    },
                )
            elif cmd == protocol.CMD_RESET:
                protocol.send_msg(
                    conn,
                    {
                        protocol.KEY_OK: True,
                        protocol.KEY_OBS: _canonical_obs(),
                        protocol.KEY_INFO: {},
                    },
                )
            elif cmd == protocol.CMD_STEP:
                action = msg["action"]
                _received["action_shape"] = tuple(action.shape)
                _received["action_dtype"] = str(action.dtype)
                protocol.send_msg(
                    conn,
                    {
                        protocol.KEY_OK: True,
                        protocol.KEY_OBS: _canonical_obs(),
                        protocol.KEY_REWARD: np.array([0.25], dtype=np.float32),
                        protocol.KEY_TERMINATED: np.array([False]),
                        protocol.KEY_TRUNCATED: np.array([False]),
                        protocol.KEY_SUCCESS: np.array([False]),
                        protocol.KEY_INFO: {},
                    },
                )
            elif cmd == protocol.CMD_CLOSE:
                protocol.send_msg(conn, {protocol.KEY_OK: True})
                return
    except (ConnectionError, EOFError):
        return
    finally:
        conn.close()


def main() -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=_fake_server, args=(srv,), daemon=True)
    t.start()

    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    # Pin shaped mode here so this stays a pure transport test (raw reward passthrough);
    # sparse/shaped reward *semantics* are covered by s1_4_reward_mode_unit.py.
    env = ArenaSocketEnv(
        env_name="FakeDrillLift",
        num_envs=NUM_ENVS,
        reward_mode="shaped",
        host="127.0.0.1",
        port=port,
        proxy_device="cpu",
        connect_timeout=15.0,
    )
    check("spec.action_dim", env.action_dim == ACTION_DIM, f"action_dim={env.action_dim}")

    obs, info = env.reset()
    check("reset.obs_keys", set(obs) == {"main_images", "states", "task_descriptions"},
          f"keys={sorted(obs)}")
    check("reset.images_numpy_uint8",
          isinstance(obs["main_images"], np.ndarray) and obs["main_images"].dtype == np.uint8
          and obs["main_images"].shape == (NUM_ENVS, *IMG_HW),
          f"{type(obs['main_images']).__name__} {getattr(obs['main_images'], 'shape', None)}")
    check("reset.states_shape",
          isinstance(obs["states"], np.ndarray) and obs["states"].shape == (NUM_ENVS, STATE_DIM),
          f"shape={getattr(obs['states'], 'shape', None)}")
    check("reset.task_descriptions", obs["task_descriptions"] == [TASK] * NUM_ENVS)

    action = torch.zeros((NUM_ENVS, ACTION_DIM), dtype=torch.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    check("step.action_on_wire_shape", _received.get("action_shape") == (NUM_ENVS, ACTION_DIM),
          f"server saw {_received.get('action_shape')}")
    check("step.action_on_wire_float32", _received.get("action_dtype") == "float32",
          f"server saw {_received.get('action_dtype')}")
    check("step.reward_is_torch_cpu",
          isinstance(reward, torch.Tensor) and reward.device.type == "cpu"
          and reward.dtype == torch.float32 and reward.shape == (NUM_ENVS,),
          f"{type(reward).__name__} {reward.dtype} {tuple(reward.shape)} dev={reward.device}")
    check("step.reward_value", float(reward[0]) == 0.25, f"reward={float(reward[0])}")
    check("step.terminated_bool_tensor",
          isinstance(terminated, torch.Tensor) and terminated.dtype == torch.bool
          and terminated.shape == (NUM_ENVS,),
          f"{terminated.dtype} {tuple(terminated.shape)}")
    check("step.truncated_bool_tensor",
          isinstance(truncated, torch.Tensor) and truncated.dtype == torch.bool)
    check("step.info_success_present", "success" in info, f"info keys={list(info)}")

    check("device.is_cpu", env.device() == torch.device("cpu"), f"device={env.device()}")

    env.close()
    check("close.no_exception", True)

    print()
    if failures:
        print(f"S1.1 UNIT FAILED ({len(failures)}): {failures}")
        return 1
    print("S1.1 UNIT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
