#!/usr/bin/env bash
# Deploy the PUBLIC Isaac-GR00T finetune env on a fresh OSMO train node (idempotent).
# Clone (pinned ab88b50) -> x86_64-only pyproject edits -> uv sync -> source patches (no eef).
# Lustre assets (base model config.json, staged backbone, dataset) persist and are handled by
# bc_finetune_drill_lift_osmo.sh, not here.
set -uo pipefail
REPO=/workspace/Isaac-GR00T
COMMIT=ab88b50c718f6528e1df9dcbaf75865d1b604760

if [ ! -d "$REPO/.git" ]; then
  echo "[deploy] cloning Isaac-GR00T (pinned $COMMIT)..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$REPO" || { echo "CLONE_FAIL"; exit 1; }
  git -C "$REPO" checkout "$COMMIT" 2>&1 | tail -2 || echo "[deploy] warn: checkout failed (using default branch)"
  git -C "$REPO" submodule update --init --recursive 2>&1 | tail -2 || true
else
  echo "[deploy] repo present, skip clone"
fi

cd "$REPO"
echo "[deploy] pyproject x86_64-only edits..."
python3 - <<'PYEDIT'
import pathlib
p = pathlib.Path("pyproject.toml")
s = p.read_text()
orig = s
s = s.replace(
'''required-environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
    "sys_platform == 'linux' and platform_machine == 'aarch64'",
]''',
'''environments = [
    "sys_platform == 'linux' and platform_machine == 'x86_64'",
]''')
s = s.replace(
'''    { path = "scripts/deployment/dgpu/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_aarch64.whl", marker = "sys_platform == 'linux' and platform_machine == 'aarch64' and python_version == '3.10'" },
''', "")
s = s.replace(
'''torchcodec = [
    { path = "scripts/deployment/dgpu/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl", marker = "sys_platform == 'linux' and platform_machine == 'aarch64'" },
]
''', "")
p.write_text(s)
print("pyproject changed" if s != orig else "pyproject UNCHANGED (already edited or anchors moved)")
PYEDIT

echo "[deploy] uv sync (slow, ~10min)..."
uv sync 2>&1 | tail -8 || { echo "UV_SYNC_FAIL"; exit 1; }

echo "[deploy] source patches (no eef; GR00T_ADD_EEF unset)..."
python3 /workspace/osmo_patch_isaac_gr00t.py "$REPO" 2>&1 | tail -20

echo "[deploy] sanity: import gr00t"
uv run python -c "import gr00t; print('gr00t import OK', gr00t.__file__)" 2>&1 | tail -2
echo "DEPLOY_DONE_OK"
