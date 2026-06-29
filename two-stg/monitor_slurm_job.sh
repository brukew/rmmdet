#!/usr/bin/env bash
# Poll a SLURM job: every 120s while pending, every 300s while running, until it leaves the queue.
#
# Usage:
#   bash two-stg/monitor_slurm_job.sh JOBID [log_file]
#   bash two-stg/monitor_slurm_job.sh   # auto-pick newest two_stg_3way for $USER

set -euo pipefail

pick_job() {
  squeue -u "$USER" -n two_stg_3way -h -o "%i" 2>/dev/null | head -1
}

JOBID="${1:-$(pick_job)}"
if [[ -z "${JOBID}" ]]; then
  echo "No job id given and no two_stg_3way job in queue for $USER." >&2
  exit 1
fi

LOG="${2:-/orcd/data/satra/001/users/brukew/actreg/two-stg/logs/monitor_job_${JOBID}.log}"
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
