"""S4.5 diagnostic — reproduce the GR00T flow-matching training loss (the wandb `actor_loss`
metric) for my public-N1.7 BC ckpt on drill_lift_rl, to compare apples-to-apples with the
internal IQL-on-Eagle run (actor_loss ~= 0.013).

Rebuilds the exact finetune pipeline (Gr00tN1d7Pipeline: same processor / normalization /
use_relative_action / horizon-50 / sharded dataset / collator), loads my ckpt via
start_from_checkpoint, and averages model(**batch)["loss"] over N batches in EVAL mode with
color-jitter + state-dropout OFF (clean-fit measurement; the IQL run also used color_jitter=0).

Run (OSMO node, Arena 4.6.5 venv, 1 free GPU):
  CUDA_VISIBLE_DEVICES=0 python diag_flow_loss.py \
    /mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000 \
    /mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl 20
"""

import os
import sys
import statistics

os.environ.setdefault("GR00T_BACKBONE_PATH", "/mnt/amlfs-07/shared/juekunl/models/nvidia/Cosmos-Reason2-2B")
os.environ.setdefault("GR00T_ACTION_HORIZON", "50")
os.environ.setdefault("GR00T_SKIP_FIRST_N_FRAMES", "1")
# single-process group (gloo -> no NCCL/shm dependency) for the rank-0 barriers in setup
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29531")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/amlfs-07/shared/juekunl/ckpts/drill_lift_bc_gr00t_n1d7/drill_lift_bc_n1d7/checkpoint-10000"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/mnt/amlfs-07/shared/datasets/isaaclab_arena_g1/drill_lift_rl"
NBATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BS = 16

import torch
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(backend="gloo")

from gr00t.configs.base_config import get_default_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model import MODEL_REGISTRY

# torchcodec is unavailable for torch 2.10 -> decode the sharded dataset video with decord.
import decord
import gr00t.data.dataset.lerobot_episode_loader as _lel


def _decord_frames(video_path, indices, decoder_kwargs=None):
    vr = decord.VideoReader(str(video_path))
    return vr.get_batch([int(i) for i in indices]).asnumpy()


_lel.get_frames_by_indices = _decord_frames

emb = EmbodimentTag.resolve("UNITREE_G1").value
config = get_default_config().load_dict(
    {"data": {"download_cache": False,
              "datasets": [{"dataset_paths": [DATA], "mix_ratio": 1.0, "embodiment_tag": emb}]}}
)
config.load_config_path = None
config.model.tune_llm = False
config.model.tune_visual = False
config.model.tune_projector = True
config.model.tune_diffusion_model = True
config.model.state_dropout_prob = 0.0
config.model.color_jitter_params = {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0}
config.model.load_bf16 = False
config.model.reproject_vision = False
config.model.model_name = os.environ["GR00T_BACKBONE_PATH"]
config.model.backbone_trainable_params_fp32 = True
config.model.use_relative_action = True
config.model.action_horizon = int(os.environ["GR00T_ACTION_HORIZON"])
config.training.experiment_name = "flowloss_probe"
config.training.start_from_checkpoint = CKPT
config.training.skip_weight_loading = False
config.training.num_gpus = 1
config.training.global_batch_size = BS
config.training.use_wandb = False
config.data.shard_size = 1024
config.data.num_shards_per_epoch = 100000
config.data.episode_sampling_rate = 0.1
config.data.skip_first_n_frames = int(os.environ["GR00T_SKIP_FIRST_N_FRAMES"])
config.validate()
print("[flowloss] config built", flush=True)

import pathlib
import tempfile

save_cfg = pathlib.Path(tempfile.mkdtemp()) / "experiment_cfg"
save_cfg.mkdir(parents=True, exist_ok=True)
pipe = MODEL_REGISTRY.get(type(config.model))(config, save_cfg)
pipe.setup()
model = pipe.return_model().eval().cuda()
train_ds = pipe.return_dataset()[0]
collator = pipe.return_collator()
print("[flowloss] pipeline ready (ShardedMixtureDataset is iterable)", flush=True)


def to_cuda(b):
    if torch.is_tensor(b):
        return b.cuda()
    if isinstance(b, dict):
        return {k: to_cuda(v) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_cuda(v) for v in b)
    return b


def get_loss(out):
    if isinstance(out, dict):
        return out.get("loss")
    if hasattr(out, "loss"):
        return out.loss
    if isinstance(out, (list, tuple)):
        return out[0]
    return None


import itertools

losses = []
data_iter = iter(train_ds)
for bi in range(NBATCH):
    buf = list(itertools.islice(data_iter, BS))
    if len(buf) < BS:  # iterable exhausted -> restart
        data_iter = iter(train_ds)
        buf += list(itertools.islice(data_iter, BS - len(buf)))
    batch = collator(buf)
    if bi == 0:
        print("[flowloss] batch keys:", list(batch.keys()), flush=True)
    batch = to_cuda(batch)
    with torch.no_grad():
        out = model(**batch)
    loss = float(get_loss(out))
    losses.append(loss)
    print(f"[flowloss] batch {bi} loss={loss:.5f}", flush=True)

_ckpt_name = CKPT.rstrip("/").rsplit("/", 1)[-1]
print(f"FLOW_LOSS_RESULT mean={statistics.mean(losses):.5f} std={statistics.pstdev(losses):.5f} "
      f"n={len(losses)} bs={BS} ckpt={_ckpt_name}", flush=True)
