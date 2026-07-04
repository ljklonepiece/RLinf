#!/usr/bin/env bash
# sync_rlinf.sh -- ship the latest local RLinf working tree to a persistent
# Lustre checkout on an OSMO cluster, the gr00t `sync_groot.sh` way: the venv
# (deps) lives persistently on Lustre and is built once; the *code* is refreshed
# on every launch so each run uses the latest local RLinf.
#
# Why this shape (and not plain rsync):
#   - this dev box does NOT mount Lustre, and the OSMO data buckets are read-only,
#     so we cannot rsync/upload directly;
#   - the base RLinf is therefore a normal `git clone` ON the cluster (github is
#     reachable from the eval nodes), pinned to our local base commit;
#   - only the local working-tree delta (`git ls-files -mo --exclude-standard`,
#     which excludes .venv/.git/__pycache__) is shipped. It is tar+base64'd and
#     streamed over osmo_exec in <=CHUNK pieces, because the exec URI caps at
#     ~12 KB (16 KB -> HTTP 414, 8 KB -> OK) and `script` caps a single arg at
#     ~128 KB. An md5 over the reassembled payload guards against a dropped chunk.
#
# The .venv (gitignored) is never in the delta, so repeated syncs leave the
# persistent venv untouched -- exactly gr00t's "persistent dependency_dir +
# disposable light repo" split.
#
# Limitation: a *delete* of a tracked file locally is not propagated (the file
# stays on the remote). For a clean slate, `rm -rf $RLINF_REMOTE` and re-run.
#
# Usage:   ./sync_rlinf.sh [CLUSTER_ID] [TASK]
# Env:     RLINF_LOCAL  (default /home/juekunl/Work/RLinf)
#          RLINF_REMOTE (default /mnt/amlfs-07/shared/juekunl/RLinf)
#          CHUNK        (default 8000 b64 bytes/chunk)
set -euo pipefail

CLUSTER="${1:-eval_ray_cluster_1n_uuid_4226-1}"
TASK="${2:-master}"
RLINF_LOCAL="${RLINF_LOCAL:-/home/juekunl/Work/RLinf}"
RLINF_REMOTE="${RLINF_REMOTE:-/mnt/amlfs-07/shared/juekunl/RLinf}"
CHUNK="${CHUNK:-8000}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSMO="$HERE/osmo_exec.sh"
TMP="$HERE/.sync_tmp"
mkdir -p "$TMP"

cd "$RLINF_LOCAL"
BASE_COMMIT="$(git rev-parse HEAD)"
echo "[sync] $RLINF_LOCAL @ $(git rev-parse --short HEAD) -> $CLUSTER:$RLINF_REMOTE"

# 0. one-time: ensure a base checkout exists on Lustre, pinned to our base commit
echo "[sync] ensuring base checkout (clone if missing)..."
OUT="$("$OSMO" "$CLUSTER" "$TASK" "
set -e
if [ ! -e $RLINF_REMOTE/.git ]; then
  echo BOOTSTRAP_CLONE
  rm -rf $RLINF_REMOTE
  git clone --quiet https://github.com/RLinf/RLinf $RLINF_REMOTE
  git -C $RLINF_REMOTE checkout --quiet $BASE_COMMIT 2>/dev/null || echo 'WARN base commit not in clone; staying on default HEAD'
fi
echo BASE_AT \$(git -C $RLINF_REMOTE rev-parse --short HEAD)
" 2>&1 || true)"
echo "$OUT" | grep -qE 'BASE_AT' || { echo "$OUT" | tail -8; echo "[sync] base bootstrap FAILED"; exit 1; }
echo "$OUT" | grep -E 'BOOTSTRAP_CLONE|WARN|BASE_AT' | sed 's/^/[sync]   /'

# 1. build working-tree delta tarball
git ls-files -mo --exclude-standard > "$TMP/files.txt"
N=$(wc -l < "$TMP/files.txt")
if [ "$N" -eq 0 ]; then echo "[sync] no local changes; nothing to ship"; exit 0; fi
tar czf "$TMP/delta.tgz" -T "$TMP/files.txt"
base64 -w0 "$TMP/delta.tgz" > "$TMP/delta.b64"
LMD5="$(md5sum "$TMP/delta.b64" | awk '{print $1}')"
echo "[sync] $N files, $(wc -c < "$TMP/delta.b64") b64 bytes, md5=$LMD5"

# 2. reset remote staging file
"$OSMO" "$CLUSTER" "$TASK" "mkdir -p $RLINF_REMOTE && : > /tmp/rlinf_delta.b64" >/dev/null 2>&1

# 3. stream chunks in order (split -d => lexically sorted chunk_0000, chunk_0001, ...)
rm -f "$TMP"/chunk_*
split -b "$CHUNK" -d -a 4 "$TMP/delta.b64" "$TMP/chunk_"
total=$(ls "$TMP"/chunk_* | wc -l)
i=0
for c in "$TMP"/chunk_*; do
  i=$((i+1))
  P="$(cat "$c")"
  "$OSMO" "$CLUSTER" "$TASK" "printf '%s' '$P' >> /tmp/rlinf_delta.b64" >/dev/null 2>&1
  printf '\r[sync] chunk %d/%d' "$i" "$total"
done
printf '\n'

# 4. verify md5, extract over the checkout, clean pycache
OUT="$("$OSMO" "$CLUSTER" "$TASK" "
RMD5=\$(md5sum /tmp/rlinf_delta.b64 | awk '{print \$1}')
if [ \"\$RMD5\" != \"$LMD5\" ]; then echo \"SYNC_FAIL md5 local=$LMD5 remote=\$RMD5\"; exit 1; fi
base64 -d /tmp/rlinf_delta.b64 | tar xzf - -C $RLINF_REMOTE
find $RLINF_REMOTE -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
echo SYNC_OK files=\$(tar tzf <(base64 -d /tmp/rlinf_delta.b64) 2>/dev/null | wc -l)
" 2>&1 || true)"
if echo "$OUT" | grep -qE 'SYNC_OK'; then
  echo "[sync] $(echo "$OUT" | grep -oE 'SYNC_OK.*')"
  echo "[sync] done -> $RLINF_REMOTE"
else
  echo "$OUT" | tail -8
  echo "[sync] FAILED"; exit 1
fi
