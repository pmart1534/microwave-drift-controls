#!/bin/bash
# =============================================================================
# compile_drift.sh
#
# Builds ONLY batch_sweep_drift from this folder, using the same Keysight
# libraries as compile_with_batch.sh in the parent folder.
#
# Usage:  ./compile_drift.sh
# Output: ./batch_sweep_drift
# =============================================================================

chmod +x *.sh 2>/dev/null

gcc batch_sweep_drift.c /usr/local/lib/libMN7021aApp.so \
    -lxml2 -lm -lrt -lpthread -fopenmp \
    -o batch_sweep_drift

if [ $? -eq 0 ]; then
  echo "Built ./batch_sweep_drift"
else
  echo "BUILD FAILED"
  exit 1
fi
