#!/usr/bin/env bash
# Third part of the review experiments (after run_review2.sh): search-seed variance on a fixed pool.
# Progress: results/review3_pipeline.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/review3_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/review3_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
echo "$(date +%H:%M) waiting for run_review2.sh" >> "$LOG"
until grep -q "ALL DONE" results/review2_pipeline.log 2>/dev/null; do sleep 60; done
step python experiments/search_seed_variance.py --config configs/final/joint_ciciiot2025.yaml
step python experiments/search_seed_variance.py --config configs/final/joint_ciciot2023.yaml
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
