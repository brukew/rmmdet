#!/usr/bin/env bash
# Poll a SLURM job: every 120s while pending, every 300s while running, until it leaves the queue.
#
# Usage:
#   bash two-stg/monitor_slurm_job.sh JOBID [log_file]
#   bash two-stg/monitor_slurm_job.sh   # auto-pick newest two_stg_3way for $USER

# --- Portable repo-root resolution (auto-inserted) --------------------------
# Locate the repo root (the directory containing paths.py) so this script runs
# from any clone name/location, under `bash` or `sbatch`. SLURM copies the
# script to a spool dir, so if the script path does not resolve we fall back to
# $SLURM_SUBMIT_DIR then $PWD. Override by exporting REPO_ROOT before launch.
_rmm_find_root() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "/" ]; do
    if [ -f "$d/paths.py" ]; then printf '%s\n' "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT:-x}/paths.py" ]; then
  REPO_ROOT="$(_rmm_find_root "$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" 2>/dev/null && pwd)")" \
    || REPO_ROOT="$(_rmm_find_root "${SLURM_SUBMIT_DIR:-$PWD}")" \
    || REPO_ROOT="$(_rmm_find_root "$PWD")" || true
fi
if [ -z "${REPO_ROOT:-}" ] || [ ! -f "${REPO_ROOT}/paths.py" ]; then
  echo "ERROR: cannot locate repo root (paths.py). cd to the repo or export REPO_ROOT." >&2
  exit 1
fi
export REPO_ROOT
# --- end repo-root resolution ----------------------------------------------

set -euo pipefail

pick_job() {
  squeue -u "$USER" -n two_stg_3way -h -o "%i" 2>/dev/null | head -1
}

JOBID="${1:-$(pick_job)}"
if [[ -z "${JOBID}" ]]; then
  echo "No job id given and no two_stg_3way job in queue for $USER." >&2
  exit 1
fi

LOG="${2:-${REPO_ROOT}/two-stg/logs/monitor_job_${JOBID}.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "======== $(date -Is) ========"
echo "Monitoring job $JOBID (pending: 120s interval, running: 300s interval)"
echo "Log: $LOG"

while true; do
  if ! line=$(squeue -j "$JOBID" -h -o "%t %M %R" 2>/dev/null | head -1) || [[ -z "$line" ]]; then
    echo "$(date -Is) Job $JOBID no longer in queue (finished / cancelled / unknown)."
    sacct -j "$JOBID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,NodeList -n 2>/dev/null | head -20 || true
    break
  fi
  st=$(echo "$line" | awk '{print $1}')
  echo "$(date -Is) $JOBID  state=$st  $line"
  case "$st" in
    R|CG|CF)
      sleep 300
      ;;
    *)
      sleep 120
      ;;
  esac
done

echo "$(date -Is) Monitor exiting."
