# shellcheck shell=bash
# ---------------------------------------------------------------------------
# cluster_env.sh — single source of truth for cluster-specific paths.
#
# Source this at the top of every launcher/helper script:
#     source "<repo>/env/cluster_env.sh"
#     mythos_activate_env          # optional: activate the python env
#
# It auto-detects the cluster from the hostname and exports MYTHOS_* anchors.
# Override detection with: export MYTHOS_CLUSTER=schmidt before sourcing.
#
# Adding another cluster = add one case arm below. Never hardcode a
# cluster-specific path anywhere else (scripts or docs) — reference the
# MYTHOS_* vars / docs/CLUSTERS.md instead.
#
# Mirrors the pattern from the sibling repo physics-fm-pretraining
# (env/cluster_env.sh), with the MYTHOS_* prefix.
# ---------------------------------------------------------------------------

# --- repo root: derived from this file's own location (cluster-agnostic) ---
_mythos_self="${BASH_SOURCE[0]:-$0}"
MYTHOS_REPO_ROOT="$(cd "$(dirname "$_mythos_self")/.." && pwd)"
MYTHOS_LOG_DIR="$MYTHOS_REPO_ROOT/logs"

# --- cluster detection -----------------------------------------------------
if [ -z "${MYTHOS_CLUSTER:-}" ]; then
  _mythos_host="$(hostname -f 2>/dev/null || hostname)"
  case "$_mythos_host" in
    g[0-9]*.voltagepark.net|*.voltagepark.net|pw-user-*|*.parallel.works) MYTHOS_CLUSTER=schmidt ;;
    sh-*|sh[0-9]*|*.sherlock.stanford.edu)                                MYTHOS_CLUSTER=sherlock ;;
    *)
      echo "cluster_env.sh: unknown host '$_mythos_host' — set MYTHOS_CLUSTER=schmidt|sherlock" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
fi

# --- per-cluster anchors ---------------------------------------------------
case "$MYTHOS_CLUSTER" in
  schmidt)
    # Parallel Works / Voltage Park cloud HPC. RHEL, no module system.
    # uv-managed venv at <repo>/.venv (PyTorch 2.11.0+cu128, Python 3.12). uv
    # ships its own CPython, so no system-Python / module dependency.
    # NOTE: must use cu128 wheels — the node driver is CUDA 12.8, so the default
    # PyPI cu130 wheel reports CUDA unavailable. Rebuild recipe:
    # docs/CLUSTERS.md + env/requirements.schmidt.lock.
    MYTHOS_ENV_ACTIVATE="${MYTHOS_ENV_ACTIVATE:-$MYTHOS_REPO_ROOT/.venv/bin/activate}"
    MYTHOS_DATA_WORK="/data/ckpt_folders/isaacju/openmythos"        # writable working root
    MYTHOS_CKPT_DIR="$MYTHOS_DATA_WORK/checkpoints"                 # training checkpoints
    MYTHOS_DATA_DIR="$MYTHOS_DATA_WORK/hf_cache"                    # HF datasets/hub cache (HF_HOME)
    MYTHOS_SLURM_PARTITION="voltagepark"
    MYTHOS_SLURM_MODULES=""                                         # no module system
    # Files land in a shared collab tree; 0007 → 0660 files + 2770 dirs so
    # teammates can read/write (matches the sibling repo's convention).
    umask 0007
    ;;
  sherlock)
    # Stub — fill in when/if OpenMythos runs on Sherlock. Conda env there.
    MYTHOS_ENV_ACTIVATE="${MYTHOS_ENV_ACTIVATE:-}"
    MYTHOS_DATA_WORK="${MYTHOS_DATA_WORK:-}"
    MYTHOS_CKPT_DIR="${MYTHOS_CKPT_DIR:-}"
    MYTHOS_DATA_DIR="${MYTHOS_DATA_DIR:-}"
    MYTHOS_SLURM_PARTITION="${MYTHOS_SLURM_PARTITION:-}"
    MYTHOS_SLURM_MODULES="${MYTHOS_SLURM_MODULES:-}"
    echo "cluster_env.sh: sherlock anchors are stubs — populate before use" >&2
    ;;
  *)
    echo "cluster_env.sh: unsupported MYTHOS_CLUSTER='$MYTHOS_CLUSTER'" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

export MYTHOS_CLUSTER MYTHOS_REPO_ROOT MYTHOS_LOG_DIR \
       MYTHOS_ENV_ACTIVATE MYTHOS_DATA_WORK MYTHOS_CKPT_DIR MYTHOS_DATA_DIR \
       MYTHOS_SLURM_PARTITION MYTHOS_SLURM_MODULES

# --- helpers ---------------------------------------------------------------

# Activate the python env (+ load toolchain modules where a module system exists).
mythos_activate_env() {
  for _m in $MYTHOS_SLURM_MODULES; do
    command -v module >/dev/null 2>&1 && module load "$_m" 2>/dev/null || true
  done
  if [ -z "$MYTHOS_ENV_ACTIVATE" ] || [ ! -f "$MYTHOS_ENV_ACTIVATE" ]; then
    echo "mythos_activate_env: env activation script not found: '$MYTHOS_ENV_ACTIVATE'" >&2
    echo "  (on $MYTHOS_CLUSTER — build the venv per docs/CLUSTERS.md and point MYTHOS_ENV_ACTIVATE at its activate script)" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$MYTHOS_ENV_ACTIVATE"
}

# Print the resolved anchor table (debug aid): `mythos_env_summary`
mythos_env_summary() {
  cat <<EOF
MYTHOS_CLUSTER        = $MYTHOS_CLUSTER
MYTHOS_REPO_ROOT      = $MYTHOS_REPO_ROOT
MYTHOS_LOG_DIR        = $MYTHOS_LOG_DIR
MYTHOS_ENV_ACTIVATE   = $MYTHOS_ENV_ACTIVATE
MYTHOS_DATA_WORK      = $MYTHOS_DATA_WORK
MYTHOS_CKPT_DIR       = $MYTHOS_CKPT_DIR
MYTHOS_DATA_DIR       = $MYTHOS_DATA_DIR
MYTHOS_SLURM_PARTITION= $MYTHOS_SLURM_PARTITION
MYTHOS_SLURM_MODULES  = ${MYTHOS_SLURM_MODULES:-<none>}
EOF
}
