"""M2b: drive RLinf's REAL Cluster + EnvWorker (the IsaacLab entrypoint's env path)
to BUILD the Arena drill_lift env -- no GR00T checkpoint.

Mirrors what scripts/reinforcement_learning/rlinf/train.py does for the env group:
set RLINF_EXT_MODULE (-> Worker auto-calls the extension's register()), load the
config, create the Ray Cluster + HybridComponentPlacement, launch the EnvWorker group,
and call init_worker() -- which runs get_env_cls("isaaclab_arena") and builds the Arena
env inside the Ray worker. This exercises the full entrypoint -> RLinf -> Arena env
chain EXCEPT the model-gated actor/rollout (those load weights; the env worker doesn't).

Run (gear container, Arena 4.6.5 venv):
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u m2b_entrypoint_env_launch.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
# This is exactly what the IsaacLab train.py entrypoint sets so RLinf workers load the
# IsaacLab<->RLinf extension and call register() on startup.
os.environ["RLINF_EXT_MODULE"] = "isaaclab_contrib.rl.rlinf.extension"

CONFIG = "/home/juekunl/Work/arena_local_bringup/drill_lift_g1_rlinf.yaml"
os.environ["RLINF_CONFIG_FILE"] = CONFIG


def main() -> int:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(CONFIG)
    print(f"[M2b] config loaded: env_type={cfg.env.train.env_type} id={cfg.env.train.init_params.id}")

    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker

    print("[M2b] creating Ray Cluster (the entrypoint's machinery)...")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    env_placement = component_placement.get_strategy("env")
    print(f"[M2b] launching EnvWorker group (placement={env_placement})...")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    print("[M2b] env_group.init_worker() -> get_env_cls(isaaclab_arena) -> build Arena env in the Ray worker...")
    env_group.init_worker().wait()

    print("[M2b] M2b OK: RLinf EnvWorker (Ray) built the Arena env via the entrypoint path "
          "(config -> Cluster -> register() -> EnvWorker -> get_env_cls -> Arena env).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
