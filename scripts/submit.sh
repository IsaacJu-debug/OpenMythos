#!/bin/bash
# ---------------------------------------------------------------------------
# submit.sh — cluster-agnostic sbatch wrapper.
#
# Injects the cluster-specific SLURM bits (--partition, --output, --error)
# from env/cluster_env.sh so launcher scripts stay free of any
# #SBATCH --partition / --output lines (those are cluster-bound).
#
# Launcher scripts keep ONLY portable resource headers
# (--job-name, --time, --nodes, --gpus-per-node, --mem, --cpus-per-task).
#
# Usage:
#   scripts/submit.sh scripts/training/smoke_test_mythos.sh [script args...]
#   scripts/submit.sh --time=04:00:00 scripts/training/pretrain_fineweb.sh   # extra sbatch opts pass through
#
# Anything before the first non-option / .sh argument is forwarded to sbatch.
# ---------------------------------------------------------------------------
set -euo pipefail

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_self/../env/cluster_env.sh"

mkdir -p "$MYTHOS_LOG_DIR"

# Split args: leading sbatch options vs. the target script (+ its args)
sbatch_opts=()
while [ $# -gt 0 ]; do
  case "$1" in
    -*) sbatch_opts+=("$1"); shift ;;
     *) break ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "submit.sh: no target script given" >&2
  echo "usage: scripts/submit.sh [sbatch opts] <script.sh> [script args]" >&2
  exit 1
fi

target="$1"; shift
if [ ! -f "$target" ]; then
  echo "submit.sh: target script not found: $target" >&2
  exit 1
fi

echo "submit.sh: cluster=$MYTHOS_CLUSTER partition=$MYTHOS_SLURM_PARTITION log_dir=$MYTHOS_LOG_DIR"
# SLURM copies the batch script into a spool dir before running it, so a
# launcher cannot locate the repo via ${BASH_SOURCE[0]}. We pin the job's cwd
# to the repo root with --chdir and rely on MYTHOS_REPO_ROOT propagating through
# the default --export=ALL; launchers resolve env/cluster_env.sh from that.
exec sbatch \
  --partition="$MYTHOS_SLURM_PARTITION" \
  --chdir="$MYTHOS_REPO_ROOT" \
  --output="$MYTHOS_LOG_DIR/%x-%j.out" \
  --error="$MYTHOS_LOG_DIR/%x-%j.err" \
  ${sbatch_opts[@]+"${sbatch_opts[@]}"} \
  "$target" "$@"
