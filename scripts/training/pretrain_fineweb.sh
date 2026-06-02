#!/bin/bash
#SBATCH --job-name=mythos_pretrain
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --cpus-per-task=64
#
# OpenMythos pretraining on FineWeb-Edu (single node, all GPUs via torchrun + FSDP).
# Thin wrapper over the upstream training entrypoint training/3b_fine_web_edu.py.
#
# Hyperparameters (model size mythos_3b, subset "sample-10BT", seq_len, step
# count) are currently hardcoded in training/3b_fine_web_edu.py (~lines 374-386,
# 395). We intentionally do NOT edit that upstream-tracked file here, to keep
# `git merge upstream/main` clean. If tunable knobs are needed, prefer
# upstreaming env-var overrides over forking the script.
#
# Cluster-specific paths come from env/cluster_env.sh (no hardcoded paths here).
# --partition / --output are injected by scripts/submit.sh.
#
# Usage:
#   scripts/submit.sh scripts/training/pretrain_fineweb.sh
#   scripts/submit.sh --time=24:00:00 --gpus-per-node=8 scripts/training/pretrain_fineweb.sh

set -euo pipefail

# Resolve env/cluster_env.sh robustly: under SLURM the script is spooled, so
# ${BASH_SOURCE[0]} is not in the repo — prefer MYTHOS_REPO_ROOT (exported by
# submit.sh, also the job's --chdir), fall back to this file's location when run
# directly (e.g. inside an salloc allocation).
if [ -n "${MYTHOS_REPO_ROOT:-}" ] && [ -f "$MYTHOS_REPO_ROOT/env/cluster_env.sh" ]; then
  _envsh="$MYTHOS_REPO_ROOT/env/cluster_env.sh"
else
  _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _envsh="$_here/../../env/cluster_env.sh"
fi
# shellcheck disable=SC1090
source "$_envsh"
mythos_activate_env

cd "$MYTHOS_REPO_ROOT"

# Route HF datasets/hub cache to the writable working root (not $HOME).
export HF_HOME="$MYTHOS_DATA_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$MYTHOS_DATA_DIR" "$MYTHOS_CKPT_DIR"

# The training script writes checkpoints to ./checkpoints by default; point it
# at the cluster ckpt dir via a symlink-free env if the script supports it.
# (As of upstream, ckpt_dir is derived internally; see training/3b_fine_web_edu.py.)

NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "pretrain_fineweb: launching torchrun with $NGPU GPU(s) | HF_HOME=$HF_HOME"

torchrun --standalone --nproc_per_node="$NGPU" \
  training/3b_fine_web_edu.py
