#!/usr/bin/env bash
# Push local file(s) to an OSMO node via chunked base64 streaming over osmo_exec
# (the exec URI caps ~12KB, so we split into <=CHUNK pieces and append, then md5-verify).
#
# Usage: push_to_node.sh <cluster> <task> LOCAL1:REMOTE1 [LOCAL2:REMOTE2 ...]
set -euo pipefail
CLUSTER="${1:?cluster}"; TASK="${2:?task}"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OSMO="$HERE/osmo_exec.sh"
CHUNK="${CHUNK:-7000}"

for pair in "$@"; do
  LOCAL="${pair%%:*}"; REMOTE="${pair#*:}"
  [ -f "$LOCAL" ] || { echo "MISSING $LOCAL"; exit 1; }
  b64="$(base64 -w0 "$LOCAL")"
  LMD5="$(printf '%s' "$b64" | md5sum | awk '{print $1}')"
  "$OSMO" "$CLUSTER" "$TASK" ": > /tmp/push.b64" >/dev/null 2>&1
  tmp="$(mktemp -d)"; printf '%s' "$b64" | split -b "$CHUNK" -d -a 4 - "$tmp/c_"
  n=0
  for c in "$tmp"/c_*; do
    n=$((n+1)); P="$(cat "$c")"
    "$OSMO" "$CLUSTER" "$TASK" "printf '%s' '$P' >> /tmp/push.b64" >/dev/null 2>&1
  done
  rm -rf "$tmp"
  OUT="$("$OSMO" "$CLUSTER" "$TASK" "RMD5=\$(md5sum /tmp/push.b64 | awk '{print \$1}'); if [ \"\$RMD5\" != \"$LMD5\" ]; then echo PUSH_FAIL_MD5 local=$LMD5 remote=\$RMD5; exit 1; fi; mkdir -p \$(dirname '$REMOTE'); base64 -d /tmp/push.b64 > '$REMOTE'; echo PUSHED \$(wc -c < '$REMOTE')B chunks=$n -> $REMOTE" 2>&1 || true)"
  echo "$OUT" | grep -E 'PUSHED|PUSH_FAIL' | tail -1 || echo "  [$REMOTE] verify line not captured (check manually)"
done
