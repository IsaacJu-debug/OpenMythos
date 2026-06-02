# State — cluster

_Last reconciled: 2026-06-02 (Entry 001)_

## Pipeline status (live)

Keep only in-flight / not-yet-stable steps here. Retire stable ones to Archive.

| Step | Status | Detail | Entry |
| --- | --- | --- | --- |
| 1-GPU smoke test | ✅ done | job 9143, H100, `mythos_1b`, SMOKE OK | Entry 001 |
| FineWeb-Edu pretrain | ⏳ not started | torchrun + FSDP, multi-GPU untested | Entry 001 |

## Findings (verified)

Durable facts later sessions rely on. Each carries the entry that established it.

- Schmidt GPU nodes run CUDA 12.8 (driver `12080`); torch must be the **cu128**
  wheel — the default cu130 wheel reports `CUDA available: False`. Lock pins
  `torch==2.11.0+cu128`. — Entry 001
- Smoke `mythos_1b` preset = 1,064,027,938 params; bf16 fwd+loss on one H100 peaks
  at 21.42 GB (ample 80 GB headroom). Untrained loss 10.7441 ≈ ln(32000). — Entry 001
- Active partition is `voltagepark`; writable root `MYTHOS_DATA_WORK` =
  `/data/ckpt_folders/isaacju/openmythos`. — Entry 001

## Known-bad paths

What broke and why, so it isn't rediscovered. Failures are reusable assets.

- Smoke job dies at startup with `cluster_env.sh: No such file or directory` →
  launcher sourced it via a relative path that breaks under the SLURM spool dir →
  resolve via `MYTHOS_REPO_ROOT` (fixed; was job 9141). — Entry 001
- `CUDA available: False` / "driver too old (12080)" → venv had cu130 torch →
  reinstall with `--torch-backend=cu128` (fixed; was job 9142). — Entry 001
- `huggingface-cli` binary gone in `huggingface_hub` ≥1.x → use `hf` CLI / Python
  API. — Entry 001

## Open routes / next

The short list of what to do next. Each points at the entry that opened it.

- [ ] Launch FineWeb-Edu pretrain (torchrun + FSDP); validate multi-GPU path — opened in Entry 001
- [ ] Commit working-tree setup (`env/`, `scripts/`, `docs/`, `CLAUDE.md`) — opened in Entry 001
- [ ] Populate Sherlock arm of `cluster_env.sh` if/when needed (stub) — opened in Entry 001

## Archive (completed & stable)

Retired pipeline rows and closed routes. Collapsed; not deleted.

- uv venv built (PyTorch 2.11.0+cu128, Py 3.12), lock frozen — Entry 001
