#!/bin/bash
# Run 3 back-to-back drift-test sessions of batch_sweep.
#
# Between sessions, waits INTER_SESSION_DELAY_SEC so the VNA has time
# to settle (simulates real "operator setup" gap between sessions).
#
# Assumes the batch_sweep binary is built and expects:
#   - Model / grid / antenna / object are all in the saved-config lists
#     (batch_configs/*.list) so the binary can be launched with an
#     answers-file to auto-select them.
#
# Usage:
#   ./run_drift_3sessions.sh [inter_session_delay_seconds]
#
# Example:
#   ./run_drift_3sessions.sh 60      # 60s pause between sessions
#
# The three sessions each produce their own timestamped folder under ./Data/
# and can be analyzed as three independent drift observations.

set -e

BINARY="${BATCH_SWEEP_BIN:-./batch_sweep}"
N_SESSIONS="${N_SESSIONS:-3}"
INTER_SESSION_DELAY_SEC="${1:-60}"

# The batch_sweep binary is INTERACTIVE for the initial setup (VNA connect,
# gain, calibration).  For drift mode, once those prompts are answered, the
# rest is fully automated.  For a truly unattended run, you can pipe answers
# in via a file - see the sample "drift_answers.txt" you might need to
# create per your specific setup.
#
# If you have not created an answers file, the script will just launch the
# binary and you'll answer the initial VNA prompts manually for each session.
# After each session's manual setup, it enters drift mode and runs unattended.

ANSWERS_FILE="${ANSWERS_FILE:-}"   # e.g., "drift_answers.txt"

echo "======================================================================"
echo "  RUN DRIFT TESTS  --  ${N_SESSIONS} sessions, ${INTER_SESSION_DELAY_SEC}s between"
echo "======================================================================"

for i in $(seq 1 $N_SESSIONS); do
  echo ""
  echo "----------------------------------------------------------------------"
  echo "  Session $i of $N_SESSIONS starting at $(date +'%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------------------------------------"

  if [ -n "$ANSWERS_FILE" ] && [ -f "$ANSWERS_FILE" ]; then
    "$BINARY" < "$ANSWERS_FILE"
  else
    "$BINARY"
  fi

  echo ""
  echo "  Session $i complete at $(date +'%Y-%m-%d %H:%M:%S')"

  if [ $i -lt $N_SESSIONS ]; then
    echo "  Waiting ${INTER_SESSION_DELAY_SEC}s before next session..."
    sleep "$INTER_SESSION_DELAY_SEC"
  fi
done

echo ""
echo "======================================================================"
echo "  ALL ${N_SESSIONS} DRIFT-TEST SESSIONS COMPLETE"
echo "  Data folders under ./Data/  (three timestamped subfolders)"
echo "======================================================================"
