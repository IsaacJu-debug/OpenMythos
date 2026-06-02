#!/bin/bash
#SBATCH --job-name=mythos_smoke
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#
# OpenMythos single-GPU smoke test.
# Verifies, on a real GPU, the full chain that pretraining depends on:
#   imports → CUDA wheel → model instantiation → forward → cross-entropy
#   → backward → AdamW step, on random token ids (no HF network needed).
#
# Attention rides on PyTorch's native SDPA (no flash-attn) — that is the
# intended path on Schmidt; see docs/CLUSTERS.md.
#
# Cluster-specific paths come from env/cluster_env.sh (no hardcoded paths here).
# --partition / --output are injected by scripts/submit.sh.
#
# Usage:
#   - inside an existing salloc allocation: bash scripts/training/smoke_test_mythos.sh
#   - via SLURM:                            scripts/submit.sh scripts/training/smoke_test_mythos.sh

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

python - <<'PY'
import torch, torch.nn as nn
from open_mythos import OpenMythos
from open_mythos.variants import mythos_1b

print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no CUDA device visible to the job"
dev = "cuda"
print(f"device: {torch.cuda.get_device_name(0)} | bf16: {torch.cuda.is_bf16_supported()}")

# Keep it small & fast: short context, fewer loop iters than the 1B default.
cfg = mythos_1b()
cfg.max_seq_len = 512
cfg.max_loop_iters = 4
model = OpenMythos(cfg).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

B, T = 2, 256
x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

model.train()
with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
loss.backward()
opt.step()
opt.zero_grad(set_to_none=True)

print(f"logits: {tuple(logits.shape)} | loss: {loss.item():.4f}")
assert torch.isfinite(loss).item(), "loss is NaN/Inf"
print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
print("SMOKE OK")
PY
