# Dev journal — cluster

Append-only history. Numbered entries, newest at the bottom. Never edit a past
entry. Current truth lives in `docs/state_cluster.md`, not here.

See `docs/INDEX.md` for the full doc map.

---

## Entry 001 — Initial repo setup on Schmidt: env + green smoke test (2026-06-02) [cluster]

**Continued from:** —  ·  **Closes:** —

### TL;DR
Stood up the fork on the Schmidt cluster: uv venv (PyTorch 2.11.0+cu128, Py
3.12), centralized cluster paths in `env/cluster_env.sh`, and got a 1-GPU SLURM
smoke test green on an H100 (job 9143) after two failed attempts.

### Problem / Goal
First-time setup of `IsaacJu-debug/OpenMythos` to train on Schmidt (Parallel
Works / Voltage Park). Needed a reproducible env, cluster paths that stay
merge-clean from upstream, and a passing single-GPU smoke test before any real
pretrain.

### What changed
- `env/cluster_env.sh` — auto-detects schmidt by hostname; exports `MYTHOS_*`
  anchors (`REPO_ROOT`, `DATA_WORK=/data/ckpt_folders/isaacju/openmythos`,
  `CKPT_DIR`, `DATA_DIR`/`HF_HOME`, `SLURM_PARTITION=voltagepark`); `umask 0007`.
- `env/requirements.schmidt.lock` — frozen deps (74 lines), pins
  `torch==2.11.0+cu128` + `nvidia-*-cu12`.
- `scripts/submit.sh` — sbatch wrapper injecting `--partition`/`--output`.
- `scripts/training/smoke_test_mythos.sh`, `scripts/training/pretrain_fineweb.sh`.
- `docs/CLUSTERS.md`, `docs/datasets.md`, `docs/open_mythos.md`, `docs/INDEX.md`,
  `CLAUDE.md`.

### Root cause
Two smoke failures before green:
- **9141** — launcher sourced `cluster_env.sh` via a relative path that didn't
  resolve under the SLURM spool dir (`.../job09141/../../env/cluster_env.sh: No
  such file or directory`). Fixed by resolving via `MYTHOS_REPO_ROOT`.
- **9142** — venv had `torch==2.11.0+cu130` (default PyPI wheel); H100 node
  driver is CUDA 12.8, so torch reported `CUDA available: False` ("driver too
  old, found version 12080"). Rebuilt with `--torch-backend=cu128`.

### Findings
- The cu128 pin is load-bearing: Schmidt GPU nodes run CUDA 12.8; the default
  cu130 torch wheel sees no device. Always install `--torch-backend=cu128`; the
  lock already pins it.
- Smoke uses the `mythos_1b` preset: 1,064,027,938 params, fwd+loss in bf16 on
  one H100, peak 21.42 GB — large headroom on 80 GB.
- Untrained loss 10.7441 ≈ ln(32000)=10.37 → sane init.

### Verification
```text
$ sacct -j 9141,9142,9143 --format=JobID,JobName,State,Elapsed,ExitCode
9141  mythos_sm+  FAILED     00:00:02  1:0
9142  mythos_sm+  FAILED     00:01:09  1:0
9143  mythos_sm+  COMPLETED  00:00:40  0:0
$ .venv/bin/python -c "import torch; print(torch.__version__)"
2.11.0+cu128
$ grep -i '^torch' env/requirements.schmidt.lock
torch==2.11.0+cu128
$ cat logs/mythos_smoke-9143.out
torch 2.11.0+cu128 | CUDA available: True
device: NVIDIA H100 80GB HBM3 | bf16: True
params: 1,064,027,938
logits: (2, 256, 32000) | loss: 10.7441
peak GPU mem: 21.42 GB
SMOKE OK
```

### Open / next
- Run the FineWeb-Edu pretrain (`scripts/submit.sh
  scripts/training/pretrain_fineweb.sh`, torchrun + FSDP) — multi-GPU path
  untested.
- Commit the working-tree setup (`env/`, `scripts/`, `docs/`, `CLAUDE.md`).
- Sherlock arm of `cluster_env.sh` is still a stub.

**Related commits:** _(uncommitted — setup still in working tree as of this entry)_
