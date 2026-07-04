#!/usr/bin/env python3
"""S1.4 unit check: sparse vs shaped reward formulation in ArenaSocketEnv.

Isolates the *reward-mode* logic (the S1.4 change) from Isaac entirely, using a tiny
thread-based fake server that speaks ``protocol`` and emits BOTH a fixed shaped reward
and a controllable ``success`` flag. It verifies the generic reward contract:

  * sparse (default): reward == success (1.0 on the success step, 0.0 else), independent
    of the dense shaped reward; ``infos["success"]`` is surfaced.
  * shaped: reward == the server's dense reward_buf, regardless of success.
  * terminated / truncated are passed through unchanged in both modes (only the reward
    channel differs).
  * fail-fast: requesting reward_mode='sparse' against a server that reports no 'success'
    term (spec.has_success=False) raises at construction (no silent all-zero reward).

The fake server makes ``success`` fire on the 2nd step after each reset, so we can see
reward flip 0.0 -> 1.0 in sparse mode while shaped stays flat.

Run with the RLinf py3.11 venv:
    /home/juekunl/Work/RLinf/.venv/bin/python \
        /home/juekunl/Work/arena_local_bringup/s1_4_reward_mode_unit.py
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
SHAPED_REWARD = 0.25
SUCCESS_ON_STEP = 2  # success fires on the 2nd step after reset


def _canonical_obs() -> dict:
    return {
        "main_images": np.zeros((NUM_ENVS, *IMG_HW), dtype=np.uint8),
        "states": np.zeros((NUM_ENVS, STATE_DIM), dtype=np.float32),
        "task_descriptions": ["lift drill"] * NUM_ENVS,
    }


def _serve_forever(sock: socket.socket, has_success: bool) -> None:
    """Accept connections one at a time; per-connection step counter."""
    while True:
        try:
            conn, _ = sock.accept()
        except OSError:
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        step_count = 0
        try:
            while True:
                msg = protocol.recv_msg(conn)
                cmd = msg.get("cmd")
                if cmd == protocol.CMD_SPEC:
                    protocol.send_msg(conn, {
                        protocol.KEY_OK: True,
                        protocol.KEY_SPEC: {
                            "env_name": "FakeLift", "num_envs": NUM_ENVS,
                            "action_dim": ACTION_DIM, "has_success": has_success,
                        },
                    })
                elif cmd == protocol.CMD_RESET:
                    step_count = 0
                    protocol.send_msg(conn, {
                        protocol.KEY_OK: True, protocol.KEY_OBS: _canonical_obs(),
                        protocol.KEY_INFO: {},
                    })
                elif cmd == protocol.CMD_STEP:
                    step_count += 1
                    success = has_success and (step_count >= SUCCESS_ON_STEP)
                    protocol.send_msg(conn, {
                        protocol.KEY_OK: True, protocol.KEY_OBS: _canonical_obs(),
                        protocol.KEY_REWARD: np.array([SHAPED_REWARD], dtype=np.float32),
                        protocol.KEY_TERMINATED: np.array([bool(success)]),
                        protocol.KEY_TRUNCATED: np.array([False]),
                        protocol.KEY_SUCCESS: np.array([bool(success)]),
                        protocol.KEY_INFO: {},
                    })
                elif cmd == protocol.CMD_CLOSE:
                    protocol.send_msg(conn, {protocol.KEY_OK: True})
                    break
        except (ConnectionError, EOFError):
            pass
        finally:
            conn.close()


def _start_server(has_success: bool) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    threading.Thread(target=_serve_forever, args=(srv, has_success), daemon=True).start()
    return srv.getsockname()[1]


def main() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    # --- server WITH a success term ---------------------------------------
    port = _start_server(has_success=True)

    # sparse: reward must equal success (0.0 then 1.0), not the shaped 0.25.
    env = ArenaSocketEnv(env_name="FakeLift", num_envs=NUM_ENVS, reward_mode="sparse",
                         host="127.0.0.1", port=port, proxy_device="cpu", connect_timeout=15.0)
    check("sparse.has_success_detected", env.has_success is True)
    env.reset()
    _, r1, term1, trunc1, info1 = env.step(torch.zeros((NUM_ENVS, ACTION_DIM)))
    _, r2, term2, trunc2, info2 = env.step(torch.zeros((NUM_ENVS, ACTION_DIM)))
    check("sparse.step1_reward_0", float(r1[0]) == 0.0, f"r1={float(r1[0])}")
    check("sparse.step2_reward_1", float(r2[0]) == 1.0, f"r2={float(r2[0])}")
    check("sparse.reward_eq_success",
          float(r1[0]) == float(info1["success"][0].float())
          and float(r2[0]) == float(info2["success"][0].float()))
    check("sparse.terminated_tracks_success",
          bool(term1[0]) is False and bool(term2[0]) is True,
          f"term1={bool(term1[0])} term2={bool(term2[0])}")
    check("sparse.truncated_false", bool(trunc1[0]) is False and bool(trunc2[0]) is False)
    check("sparse.info_success_tensor",
          isinstance(info2["success"], torch.Tensor) and info2["success"].dtype == torch.bool)
    env.close()

    # shaped: reward must equal the dense 0.25 regardless of success.
    env = ArenaSocketEnv(env_name="FakeLift", num_envs=NUM_ENVS, reward_mode="shaped",
                         host="127.0.0.1", port=port, proxy_device="cpu", connect_timeout=15.0)
    env.reset()
    _, r1, _, _, _ = env.step(torch.zeros((NUM_ENVS, ACTION_DIM)))
    _, r2, _, _, info2 = env.step(torch.zeros((NUM_ENVS, ACTION_DIM)))
    check("shaped.step1_reward_dense", abs(float(r1[0]) - SHAPED_REWARD) < 1e-6, f"r1={float(r1[0])}")
    check("shaped.step2_reward_dense", abs(float(r2[0]) - SHAPED_REWARD) < 1e-6, f"r2={float(r2[0])}")
    check("shaped.info_success_present_on_success", bool(info2["success"][0]) is True,
          "shaped mode still surfaces the true success flag")
    env.close()

    # --- server WITHOUT a success term: sparse must fail fast --------------
    port_ns = _start_server(has_success=False)
    raised = False
    try:
        ArenaSocketEnv(env_name="NoSuccessEnv", num_envs=NUM_ENVS, reward_mode="sparse",
                       host="127.0.0.1", port=port_ns, proxy_device="cpu", connect_timeout=15.0)
    except RuntimeError as e:
        raised = "has_success" in str(e) or "success" in str(e)
    check("nosuccess.sparse_raises", raised, "sparse without success term -> RuntimeError")

    # shaped against the same server must still work (no success term needed).
    env = ArenaSocketEnv(env_name="NoSuccessEnv", num_envs=NUM_ENVS, reward_mode="shaped",
                         host="127.0.0.1", port=port_ns, proxy_device="cpu", connect_timeout=15.0)
    env.reset()
    _, r1, _, _, _ = env.step(torch.zeros((NUM_ENVS, ACTION_DIM)))
    check("nosuccess.shaped_ok", abs(float(r1[0]) - SHAPED_REWARD) < 1e-6, f"r1={float(r1[0])}")
    env.close()

    # invalid mode rejected.
    bad = False
    try:
        ArenaSocketEnv(env_name="x", num_envs=NUM_ENVS, reward_mode="nope",
                       host="127.0.0.1", port=port, connect_timeout=5.0)
    except ValueError:
        bad = True
    check("invalid_mode_rejected", bad)

    print()
    if failures:
        print(f"S1.4 REWARD-MODE FAILED ({len(failures)}): {failures}")
        return 1
    print("S1.4 REWARD-MODE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
