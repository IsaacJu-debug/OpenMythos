# Clusters

> Single source of truth for cluster-specific paths is `env/cluster_env.sh`.
> **Never hardcode a cluster path** in a script or doc — reference the `MYTHOS_*`
> anchors below. Adding a cluster = add one case arm in `cluster_env.sh`.

`cluster_env.sh` auto-detects the cluster from the hostname:
`*.voltagepark.net` / `pw-user-*` / `*.parallel.works` → **schmidt**;
`sh-*` / `*.sherlock.stanford.edu` → **sherlock**. Override with
`export MYTHOS_CLUSTER=schmidt` before sourcing.

## Anchor table

| `MYTHOS_*` var | Meaning | **Schmidt** (Parallel Works / Voltage Park) | **Sherlock** (Stanford) |
|---|---|---|---|
| `MYTHOS_CLUSTER` | cluster id | `schmidt` | `sherlock` |
| `MYTHOS_REPO_ROOT` | repo root | derived from `cluster_env.sh` location | same |
| `MYTHOS_ENV_ACTIVATE` | python env activate | `$MYTHOS_REPO_ROOT/.venv/bin/activate` (uv venv, PyTorch 2.11.0, Py 3.12) | _(stub — not set up)_ |
| `MYTHOS_DATA_WORK` | writable working root | `/data/ckpt_folders/isaacju/openmythos` | _(stub)_ |
| `MYTHOS_CKPT_DIR` | training checkpoints | `$MYTHOS_DATA_WORK/checkpoints` | _(stub)_ |
| `MYTHOS_DATA_DIR` | HF cache (`HF_HOME`) | `$MYTHOS_DATA_WORK/hf_cache` | _(stub)_ |
| `MYTHOS_SLURM_PARTITION` | sbatch partition | `voltagepark` | _(stub)_ |
| `MYTHOS_SLURM_MODULES` | toolchain modules | _(none — no module system)_ | _(stub)_ |

Resolve actual values at runtime: `source env/cluster_env.sh && mythos_env_summary`.

## Schmidt (Parallel Works / Voltage Park cloud HPC)

- RHEL, **no module system**. `uv` ships its own CPython, so there is no
  system-Python dependency.
- SLURM partitions (from `sinfo`): `voltagepark` (default, 8 nodes × `gpu:8`,
  7-day limit), `cardinal` (8 × `gpu:8`), `cs321m` (1 × `gpu:8`, 8h limit).
  We use `voltagepark`.
- Python env: **uv-managed venv** at `$MYTHOS_REPO_ROOT/.venv`
  (PyTorch 2.11.0**+cu128**, Python 3.12). Frozen recipe:
  `env/requirements.schmidt.lock`.
- **⚠️ CUDA wheel must be cu128.** The GPU node driver is CUDA **12.8**
  (`found version 12080`). The default PyPI torch 2.11.0 wheel is **cu130** and
  reports `torch.cuda.is_available() == False` ("NVIDIA driver too old"). Always
  install with `--torch-backend=cu128` so torch + `nvidia-*-cu12` come from the
  PyTorch cu128 index.

### Rebuild the env from scratch

```bash
cd "$MYTHOS_REPO_ROOT"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r env/requirements.schmidt.lock
uv pip install --python .venv/bin/python -e . --no-deps   # register the open_mythos package itself
```

> `uv pip freeze` does not record the editable `open-mythos` install, so the
> lock pins only the third-party deps — the `-e . --no-deps` line adds the
> package back without re-resolving anything. The lock already pins the cu128
> torch + `nvidia-*-cu12` builds, so no `--torch-backend` is needed when
> installing from it.

Or resolve fresh from the package (then re-freeze when deps change):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cu128 -e . loguru   # cu128 wheels; loguru used by training/
uv pip freeze --python .venv/bin/python > env/requirements.schmidt.lock
```

### Notes / gotchas

- **No flash-attn.** `GQAttention`/`MLAttention` ride on PyTorch's native SDPA,
  which is competitive with FlashAttention 2 on current cards — intended path.
  Re-add the `[flash]` extra later only on a GPU node with `nvcc` if profiling
  demands it.
- **`huggingface_hub` ≥ 1.x dropped the `huggingface-cli` binary.** Use the `hf`
  CLI or the Python API; any script invoking `huggingface-cli` will break.
- Schmidt arm of `cluster_env.sh` sets `umask 0007` so files created under the
  shared `/data` tree stay group-readable/writable.

## Sherlock

Stub only — `cluster_env.sh` has a `sherlock` arm with empty anchors and a
warning. Populate it (conda env activate, Oak/scratch paths, partition) if
OpenMythos ever needs to run there.
