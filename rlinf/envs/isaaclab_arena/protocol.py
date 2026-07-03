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

"""Length-framed pickle transport for the IsaacLab-Arena env service.

This is the wire between the RLinf env worker (CPython 3.11) and the out-of-process
Arena env server (CPython 3.12). It is **stdlib-only on purpose** so it imports
cleanly in *both* interpreters without dragging in torch/numpy/rlinf: payloads carry
numpy arrays, but those are pickled by the caller, not imported here.

Framing: an 8-byte big-endian unsigned length header followed by a pickle payload
(protocol 5, supported by both 3.11 and 3.12). Messages are plain dicts; see the
``CMD_*`` / response-key constants below for the contract.

Why pickle (not msgpack/json): numpy arrays (uint8 images, float32 state) pickle
losslessly with zero extra deps, and the boundary is a trusted local process we
spawn ourselves -- never an untrusted peer.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

# 8-byte big-endian unsigned length prefix.
_HEADER = struct.Struct("!Q")
_PICKLE_PROTOCOL = 5  # supported by CPython 3.8+ (both 3.11 and 3.12)

# Command names (driver -> server). Carried under the "cmd" key.
CMD_PING = "ping"
CMD_SPEC = "spec"  # request action/obs spec (dims, dtypes) for space construction
CMD_RESET = "reset"
CMD_STEP = "step"
CMD_CLOSE = "close"

# Response keys (server -> driver).
KEY_OK = "ok"  # bool: did the command succeed
KEY_ERROR = "error"  # str: traceback when ok is False
KEY_OBS = "obs"  # dict[str, np.ndarray | list]: canonical RLinf obs
KEY_REWARD = "reward"  # np.ndarray (num_envs,): Arena's shaped per-step reward (reward_buf)
KEY_TERMINATED = "terminated"  # np.ndarray (num_envs,) bool: true terminations (success|drop)
KEY_TRUNCATED = "truncated"  # np.ndarray (num_envs,) bool: time-out only
KEY_SUCCESS = "success"  # np.ndarray (num_envs,) bool: the 'success' termination term only
KEY_INFO = "info"  # dict: sanitized, picklable extras
KEY_SPEC = "spec"  # dict: {"action_dim", "num_envs", "has_success", ...}


def send_msg(sock: socket.socket, obj: Any) -> None:
    """Pickle ``obj`` and write it to ``sock`` with a length prefix."""
    data = pickle.dumps(obj, protocol=_PICKLE_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes from ``sock`` or raise on early close."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"socket closed with {len(buf)}/{n} bytes received"
            )
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Any:
    """Read one length-framed pickle message from ``sock``."""
    (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
    return pickle.loads(_recv_exactly(sock, length))
