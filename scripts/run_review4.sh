#!/usr/bin/env bash
# Fourth part of the review experiments (after run_review3.sh): the stronger baselines (tuned two-tier cascade,
# greedy search) under the BLE energy profiles, for the energy-budget and operating-point analyses.
# Progress: results/review4_pipeline.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/review4_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/review4_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
echo "$(date +%H:%M) waiting for run_review3.sh" >> "$LOG"
until grep -q "ALL DONE" results/review3_pipeline.log 2>/dev/null; do sleep 60; done
step python experiments/cost_sensitivity.py --config configs/final/energy_baselines_ciciiot2025.yaml
step python experiments/cost_sensitivity.py --config configs/final/energy_baselines_ciciot2023.yaml
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
