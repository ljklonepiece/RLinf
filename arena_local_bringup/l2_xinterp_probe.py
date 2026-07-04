"""L2 de-risk probe: cross-interpreter Ray (RLinf py3.11 driver -> Arena py3.12 worker).

This validates the single biggest unknown for the "Option A" RLinf<->IsaacLab-Arena
integration: that a Ray driver started from the RLinf py3.11 venv can launch a Ray
worker pinned to the Arena py3.12 interpreter via ``runtime_env["py_executable"]``,
that both join the SAME Ray cluster (ray must be the same version on both -> we
aligned Arena to ray 2.55.1), and that observation-like payloads (numpy arrays,
CPU torch tensors) serialize cleanly across the py3.11 <-> py3.12 boundary.

It does NOT launch Isaac Sim (no GPU/Kit) -- that is a heavier follow-up. The point
here is to isolate the interpreter/serialization boundary in ~1 file so we don't
discover a fundamental incompatibility deep inside a full DSRL run.

Run from the RLinf venv (py3.11):
    /home/juekunl/Work/RLinf/.venv/bin/python -u \
        /home/juekunl/Work/arena_local_bringup/l2_xinterp_probe.py

Env vars:
    ARENA_PY            -> Arena venv python (default: IsaacLab-Arena/.venv/bin/python)
    ISAACLAB_ARENA_PATH -> Arena checkout (default: /home/juekunl/Work/IsaacLab-Arena)
    PROBE_IMPORT_ARENA  -> "1" to also import isaaclab_arena_environments in the worker
                           (proves cp312 extensions load; heavier, pulls Isaac deps)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import ray

ARENA_PY = os.environ.get(
    "ARENA_PY", "/home/juekunl/Work/IsaacLab-Arena/.venv/bin/python"
)
ARENA_PATH = os.environ.get("ISAACLAB_ARENA_PATH", "/home/juekunl/Work/IsaacLab-Arena")
IMPORT_ARENA = os.environ.get("PROBE_IMPORT_ARENA", "0") == "1"


@ray.remote(num_cpus=1)
class CrossInterpProbe:
    """Runs inside the Arena py3.12 interpreter (via py_executable)."""

    def info(self) -> dict:
        out = {
            "py": sys.version.split()[0],
            "executable": sys.executable,
        }
        try:
            import ray as _ray

            out["ray"] = _ray.__version__
        except Exception as e:  # noqa: BLE001
            out["ray"] = f"FAIL {e!r}"
        try:
            import torch

            out["torch"] = torch.__version__
        except Exception as e:  # noqa: BLE001
            out["torch"] = f"FAIL {e!r}"
        if IMPORT_ARENA:
            try:
                import isaaclab_arena_environments.cli  # noqa: F401

                out["isaaclab_arena_import"] = "OK"
            except Exception as e:  # noqa: BLE001
                out["isaaclab_arena_import"] = f"FAIL {type(e).__name__}: {e}"
        return out

    def echo_numpy(self, arr: np.ndarray) -> dict:
        """Receive a numpy array from the py3.11 driver, return a py3.12-made one."""
        import torch

        t = torch.from_numpy(arr)  # numpy -> CPU torch in py3.12
        doubled = (t * 2).numpy()  # back to numpy to return across the boundary
        return {
            "recv_shape": tuple(arr.shape),
            "recv_dtype": str(arr.dtype),
            "recv_sum": float(arr.sum()),
            "ret_arr": doubled,
        }

    def make_fake_obs(self) -> dict:
        """Mimic an Arena obs payload (uint8 image + float state) crossing back."""
        import torch

        img = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)
        state = torch.arange(43, dtype=torch.float32).unsqueeze(0)
        # Return CPU tensors (what _wrap_obs would hand to RLinf after .cpu()).
        return {"main_images": img, "states": state}


def main() -> int:
    print(f"[probe] driver py={sys.version.split()[0]} executable={sys.executable}")
    print(f"[probe] driver ray={ray.__version__}")
    print(f"[probe] ARENA_PY={ARENA_PY}")
    if not os.path.exists(ARENA_PY):
        print(f"[probe] FAIL: ARENA_PY does not exist: {ARENA_PY}")
        return 1

    ray.init(ignore_reinit_error=True)

    runtime_env = {
        "py_executable": ARENA_PY,
        "env_vars": {
            "ISAACLAB_ARENA_PATH": ARENA_PATH,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        },
    }
    actor = CrossInterpProbe.options(runtime_env=runtime_env).remote()

    info = ray.get(actor.info.remote())
    print("\n[probe] ==== worker info (should be py3.12, ray 2.55.1) ====")
    for k, v in info.items():
        print(f"    {k:24} = {v}")

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    echo = ray.get(actor.echo_numpy.remote(arr))
    ret = echo.pop("ret_arr")
    print("\n[probe] ==== numpy roundtrip across py3.11<->py3.12 ====")
    print(f"    sent sum={float(arr.sum())}  -> {echo}")
    print(f"    returned doubled-array sum={float(ret.sum())} (expect {float(arr.sum()) * 2})")
    roundtrip_ok = np.allclose(ret, arr * 2)

    fake = ray.get(actor.make_fake_obs.remote())
    print("\n[probe] ==== fake obs payload (uint8 img + float state) ====")
    for k, v in fake.items():
        print(f"    {k:14} type={type(v).__name__} shape={tuple(v.shape)} dtype={v.dtype}")

    worker_is_py312 = info.get("py", "").startswith("3.12")
    ray_matches = info.get("ray") == ray.__version__

    print("\n[probe] ==== VERDICT ====")
    print(f"    worker is py3.12         : {worker_is_py312}")
    print(f"    ray version matches      : {ray_matches} (worker {info.get('ray')} / driver {ray.__version__})")
    print(f"    numpy roundtrip correct  : {roundtrip_ok}")
    if IMPORT_ARENA:
        print(f"    isaaclab_arena import    : {info.get('isaaclab_arena_import')}")

    ok = worker_is_py312 and ray_matches and roundtrip_ok
    print(f"\n[probe] {'PROBE OK' if ok else 'PROBE FAILED'}")
    ray.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
