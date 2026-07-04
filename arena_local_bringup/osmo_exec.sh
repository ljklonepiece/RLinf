#!/usr/bin/env bash
# Run a one-shot command on an OSMO workflow task.
#
# `osmo workflow exec` is interactive (it queries the terminal size and fails with
# "Inappropriate ioctl for device" when there's no TTY). This wrapper provides a PTY via
# `script`, and base64-encodes the remote command so arbitrary quoting survives the
# multiple shell layers (local bash -> script -> osmo --entry -> remote bash).
#
# Usage:
#   osmo_exec.sh <workflow_id> <task> <remote bash command ...>
# Example:
#   osmo_exec.sh train_ray_cluster_2n_uuid_cb5c-1 master 'hostname; ls /mnt/amlfs-07'
set -uo pipefail

WF="${1:?workflow_id required}"
TASK="${2:?task required}"
shift 2
REMOTE_CMD="$*"
if [[ -z "${REMOTE_CMD}" ]]; then
  echo "no remote command given" >&2
  exit 2
fi

# base64 (alnum + / + =) has no shell metacharacters, so it threads cleanly through
# every quoting layer below.
B64="$(printf '%s' "${REMOTE_CMD}" | base64 -w0)"
INNER="echo ${B64} | base64 -d | bash"

exec script -qfec \
  "osmo workflow exec --connect-timeout 180 --entry \"bash -lc '${INNER}'\" '${WF}' '${TASK}'" \
  /dev/null
