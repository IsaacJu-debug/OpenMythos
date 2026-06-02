# CLAUDE.md

## Project Overview

OpenMythos — an open-source theoretical reconstruction of a Recurrent-Depth
Transformer (RDT): **Prelude** (transformer blocks) → looped **Recurrent Block**
(up to `max_loop_iters`) → **Coda**. Attention switches between MLA and GQA; the
feed-forward is a sparse MoE (routed + shared experts). Package lives in
`open_mythos/` (`main.py`, `moda.py`, `variants.py`, `tokenizer.py`); size presets
in `open_mythos/variants.py` (`mythos_1b` … `mythos_1t`).

This is a **fork** of `kyegomez/OpenMythos` set up to run on the Schmidt cluster.

## Environment Setup

Runs on the **Schmidt cluster** (Parallel Works / Voltage Park). All
cluster-specific paths are centralized in `env/cluster_env.sh` —
**never hardcode them**. Full anchor table + rebuild recipe: `docs/CLUSTERS.md`.

**Activate the env before any Python command:**
```bash
source env/cluster_env.sh && mythos_activate_env
```
Auto-detects the cluster and activates the uv venv (`.venv`, PyTorch 2.11.0+cu128,
Python 3.12). Anchors exported: `MYTHOS_REPO_ROOT`, `MYTHOS_DATA_WORK`,
`MYTHOS_CKPT_DIR`, `MYTHOS_DATA_DIR` (HF cache), `MYTHOS_SLURM_PARTITION`,
`MYTHOS_LOG_DIR`. Debug the resolved table with `mythos_env_summary`.

The venv is reproduced from `env/requirements.schmidt.lock` (see
`docs/CLUSTERS.md` for the exact `uv` commands). **No flash-attn** — attention
uses PyTorch's native SDPA, which is competitive on current GPUs.

## SLURM

Submit jobs with the wrapper, which injects `--partition` / `--output` from
`cluster_env.sh` (launcher scripts carry only portable `#SBATCH` resource lines):
```bash
scripts/submit.sh scripts/training/smoke_test_mythos.sh        # 1-GPU smoke
scripts/submit.sh scripts/training/pretrain_fineweb.sh         # FineWeb-Edu pretrain (torchrun + FSDP)
scripts/submit.sh --time=24:00:00 scripts/training/pretrain_fineweb.sh   # extra sbatch opts pass through
```
Logs land in `logs/<job-name>-<jobid>.{out,err}`.

## Keeping in sync with upstream

`origin` = this fork (`IsaacJu-debug/OpenMythos`); `upstream` =
`kyegomez/OpenMythos` (push disabled). Pull upstream changes with:
```bash
git fetch upstream
git merge upstream/main        # or: git rebase upstream/main
```
To stay merge-clean, cluster setup lives only in `env/`, `scripts/`, `docs/`,
and this `CLAUDE.md`. **Do not edit upstream-tracked source** (`open_mythos/`,
`pyproject.toml`, `training/`, `tests/`); if a knob is needed in
`training/3b_fine_web_edu.py`, prefer an env-var override worth upstreaming over
a local fork of the file.

## Git Conventions

- Never add Claude as a co-author / trailer on commits.

## Dev journal

`docs/INDEX.md` routes `/catchup` and `/log-dev`. After a work session that
produces facts (job IDs, sizes, fixes), run `/log-dev` to update the state docs.
