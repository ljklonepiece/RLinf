"""L2-B raw cross-process smoke: RLinf py3.11 client <-> Arena py3.12 socket server.

This proves the thing the Ray probe could not: a real IsaacLab-Arena observation
(rendered image + 43-d proprio state) crossing the py3.11 <-> py3.12 boundary over a
socket, plus an action going the other way and producing a reward. It validates the
``protocol`` + ``arena_server`` + ``g1_layout`` stack end-to-end with no RLinf/Ray
machinery in the way.

Procedure (two terminals, or background the server):
    # 1) Start the server (py3.12, in the parity container):
    ARENA_PORT=5557 ARENA_ENV=LMDrillLiftRlD1 ARENA_NUM_ENVS=1 \
        /home/juekunl/Work/arena_local_bringup/run_arena_server.sh

    # 2) Once arena_server.ready appears, run this client from the RLinf venv (py3.11):
    /home/juekunl/Work/RLinf/.venv/bin/python -u \
        /home/juekunl/Work/arena_local_bringup/l2b_socket_smoke.py

Env vars:
    ARENA_HOST   (default 127.0.0.1)
    ARENA_PORT   (default: read from READY_FILE if present, else 5557)
    READY_FILE   (default: arena_local_bringup/arena_server.ready)
    SMOKE_STEPS  (default 5)
"""

from __future__ import annotations

import os
import socket
import sys
import time

import numpy as np

# Import the shared protocol straight from the RLinf repo (py3.11 side).
sys.path.insert(0, "/home/juekunl/Work/RLinf/rlinf/envs/isaaclab_arena")
import protocol  # noqa: E402

HOST = os.environ.get("ARENA_HOST", "127.0.0.1")
READY_FILE = os.environ.get(
    "READY_FILE", "/home/juekunl/Work/arena_local_bringup/arena_server.ready"
)
STEPS = int(os.environ.get("SMOKE_STEPS", "5"))


def _resolve_port() -> int:
    if os.environ.get("ARENA_PORT"):
        return int(os.environ["ARENA_PORT"])
    if os.path.exists(READY_FILE):
        with open(READY_FILE) as f:
            return int(f.read().strip())
    return 5557


def _wait_ready(timeout_s: float = 600.0) -> None:
    """Block until the server ready-file exists (env build takes ~60s+)."""
    if os.environ.get("ARENA_PORT"):
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(READY_FILE):
            return
        time.sleep(2.0)
    print(f"[smoke] WARNING: ready-file {READY_FILE} never appeared", flush=True)


def main() -> int:
    _wait_ready()
    port = _resolve_port()
    print(f"[smoke] client py={sys.version.split()[0]} connecting to {HOST}:{port}", flush=True)

    sock = socket.create_connection((HOST, port), timeout=120)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    protocol.send_msg(sock, {"cmd": protocol.CMD_PING})
    print("[smoke] ping ->", protocol.recv_msg(sock), flush=True)

    protocol.send_msg(sock, {"cmd": protocol.CMD_SPEC})
    spec_resp = protocol.recv_msg(sock)
    print("[smoke] spec ->", spec_resp, flush=True)
    spec = spec_resp[protocol.KEY_SPEC]
    num_envs = spec["num_envs"]
    action_dim = spec["action_dim"]

    protocol.send_msg(sock, {"cmd": protocol.CMD_RESET})
    reset_resp = protocol.recv_msg(sock)
    if not reset_resp.get(protocol.KEY_OK):
        print("[smoke] RESET FAILED:\n", reset_resp.get(protocol.KEY_ERROR), flush=True)
        return 1
    obs = reset_resp[protocol.KEY_OBS]
    print("[smoke] ==== reset obs (crossed py3.12 -> py3.11) ====", flush=True)
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"    {k:18} ndarray shape={v.shape} dtype={v.dtype}", flush=True)
        else:
            print(f"    {k:18} {type(v).__name__} = {v}", flush=True)

    idle = np.zeros((num_envs, action_dim), dtype=np.float32)
    rewards = []
    for i in range(STEPS):
        protocol.send_msg(sock, {"cmd": protocol.CMD_STEP, "action": idle})
        step_resp = protocol.recv_msg(sock)
        if not step_resp.get(protocol.KEY_OK):
            print(f"[smoke] STEP {i} FAILED:\n", step_resp.get(protocol.KEY_ERROR), flush=True)
            return 1
        r = step_resp[protocol.KEY_REWARD]
        rewards.append(float(np.asarray(r).mean()))
        if i == 0:
            print(
                f"[smoke] step0 reward shape={r.shape} dtype={r.dtype} "
                f"terminated={step_resp[protocol.KEY_TERMINATED]} "
                f"truncated={step_resp[protocol.KEY_TRUNCATED]} "
                f"info_keys={list(step_resp[protocol.KEY_INFO].keys())}",
                flush=True,
            )
    print(f"[smoke] {STEPS} idle steps OK. mean rewards per step = {rewards}", flush=True)

    protocol.send_msg(sock, {"cmd": protocol.CMD_CLOSE})
    print("[smoke] close ->", protocol.recv_msg(sock), flush=True)
    sock.close()

    print("\n[smoke] PROBE OK: real Arena obs + reward crossed py3.11<->py3.12 socket.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
