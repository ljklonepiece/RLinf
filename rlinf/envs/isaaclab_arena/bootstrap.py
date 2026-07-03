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

"""Runtime bootstrap for hosting an IsaacLab-Arena env inside an RLinf worker.

IsaacLab-Arena (commit dd6618c68 + ``isaaclab==4.6.5``) is built for CPython 3.12
(cp312), while the RLinf core runs on CPython 3.11 (cp311). Loading the cp312 Arena
extension modules in a cp311 interpreter is an ABI violation, so the Arena env must
run under the Arena ``.venv`` interpreter.

In RLinf this is achieved with the heterogeneous-cluster feature: the env worker
group is given ``python_interpreter_path`` = ``<ISAACLAB_ARENA_PATH>/.venv/bin/python``
(py3.12), which RLinf forwards to Ray as ``runtime_env["py_executable"]``. The env
worker process therefore already runs under the Arena venv. This module then makes
the Isaac Sim *runtime* importable from that process by:

1. pulling the Isaac/Arena environment variables the venv ``activate`` script exports
   (``ISAACLAB_PATH``, ``ISAAC_PATH``, ``EXP_PATH``, ``CARB_APP_PATH``,
   ``RESOURCE_NAME``, ``LD_LIBRARY_PATH``) into ``os.environ``;
2. ``RTLD_GLOBAL``-loading ``libpython`` and the Kit shared libraries so Isaac Sim's
   carbonite plugins resolve their symbols;
3. launching the Kit ``AppLauncher``.

This mirrors the proven GR00T eval-cluster bootstrap
(``groot/vla/eval/sim/isaaclab_arena_worker.py``). The cp-ABI ``sys.path`` filtering
that bootstrap performs is intentionally omitted here: under Option A the worker
interpreter *is* the Arena venv, so its ``site-packages`` are already authoritative.
"""

from __future__ import annotations

import ctypes
import glob
import os
import subprocess
import sys
import sysconfig

# Environment variables exported by the Arena venv ``activate`` script that Isaac Sim
# needs at import time. Pulled from a sourced subshell rather than hard-coded.
_ISAAC_ENV_KEYS = (
    "ISAACLAB_PATH",
    "ISAAC_PATH",
    "EXP_PATH",
    "CARB_APP_PATH",
    "RESOURCE_NAME",
    "LD_LIBRARY_PATH",
)


def get_arena_path() -> str:
    """Resolve the IsaacLab-Arena checkout path.

    Honors ``ISAACLAB_ARENA_PATH`` (set on the env worker group's ``env_vars``);
    raises if unset, since the env cannot be built without it.
    """
    arena_path = os.environ.get("ISAACLAB_ARENA_PATH")
    if not arena_path:
        raise RuntimeError(
            "ISAACLAB_ARENA_PATH is not set. The IsaacLab-Arena env worker group must "
            "export ISAACLAB_ARENA_PATH (the Arena checkout) in its cluster env_vars."
        )
    if not os.path.isdir(arena_path):
        raise RuntimeError(f"ISAACLAB_ARENA_PATH does not exist: {arena_path}")
    return arena_path


def _pull_activate_env_vars(arena_path: str) -> None:
    """Source the Arena venv ``activate`` and copy Isaac env vars into ``os.environ``."""
    venv_activate = os.path.join(arena_path, ".venv", "bin", "activate")
    if not os.path.exists(venv_activate):
        return
    result = subprocess.run(
        ["bash", "-c", f"source {venv_activate} && env -0"],
        capture_output=True,
        text=True,
        check=False,
    )
    activated_env = dict(
        line.split("=", 1) for line in result.stdout.split("\0") if "=" in line
    )
    for key in _ISAAC_ENV_KEYS:
        if key in activated_env and activated_env[key]:
            os.environ[key] = activated_env[key]


def _preload_native_libs(arena_path: str) -> None:
    """``RTLD_GLOBAL``-load libpython and Kit libs so carbonite plugins resolve symbols."""
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    libpython = os.path.join(
        libdir,
        f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0",
    )
    if os.path.exists(libpython):
        try:
            ctypes.CDLL(libpython, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    isaac_sim_path = os.path.join(arena_path, "submodules", "IsaacLab", "_isaac_sim")
    kit_dir = os.path.join(isaac_sim_path, "kit")
    for so_file in glob.glob(os.path.join(kit_dir, "lib*.so")):
        try:
            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def launch_isaac_app(headless: bool = True, enable_cameras: bool = True):
    """Bootstrap the Arena runtime and launch the Kit app.

    Args:
        headless: Run Isaac Sim without a GUI window (always True on a cluster).
        enable_cameras: Enable the RTX camera pipeline (required for image obs).

    Returns:
        The ``simulation_app`` handle (Kit ``SimulationApp``). Keep it alive for the
        lifetime of the env and call ``.close()`` on teardown.
    """
    # Isaac Sim blocks on an EULA stdin prompt on first import; auto-accept.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")

    arena_path = get_arena_path()
    _pull_activate_env_vars(arena_path)
    _preload_native_libs(arena_path)

    from isaaclab.app import AppLauncher

    # AppLauncher inspects sys.argv; isolate it so it doesn't see RLinf/Hydra args.
    saved_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        # multi_gpu must be False: RLinf pins one GPU per env worker via VISIBLE_DEVICES.
        app_launcher = AppLauncher(
            headless=headless, enable_cameras=enable_cameras, multi_gpu=False
        )
    finally:
        sys.argv = saved_argv

    return app_launcher.app
