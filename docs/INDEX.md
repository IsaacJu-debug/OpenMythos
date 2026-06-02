# Docs index

Routing table for `/catchup` and `/log-dev`. Each `## <area>` heading lists the
doc files for that area. State docs (`state_<area>.md`) and journals
(`develop_<area>.md`) are created by `/log-dev` as work accrues.

## cluster

- [state_cluster.md](state_cluster.md) — live cluster state: pipeline status,
  verified findings, known-bad paths, open routes.
- [develop_cluster.md](develop_cluster.md) — cluster dev journal (append-only).
- [CLUSTERS.md](CLUSTERS.md) — cluster anchor table (`MYTHOS_*`), Schmidt env
  rebuild recipe, SLURM partitions, gotchas.

## model

- [open_mythos.md](open_mythos.md) — architecture notes (RDT: Prelude → Recurrent
  Block → Coda; MLA/GQA; MoE).
- [datasets.md](datasets.md) — recommended training datasets (FineWeb-Edu, etc.).
