#!/usr/bin/env bash
# Experiments added after the internal review: dataset-specific record size for CIC IIoT 2025, and the
# temporal and held-out-device protocols (pool, tuned server models, joint search, energy profiles).
# Progress: results/review_pipeline.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/review_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/review_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
step python experiments/cost_sensitivity.py --config configs/final/cost_ciciiot2025.yaml
for m in temporal device; do
  step python experiments/build_pool.py --config configs/experiments/pool_ciciiot2025_$m.yaml
  step python experiments/build_pool.py --config configs/experiments/pool_strong_ciciiot2025_$m.yaml
  step python experiments/joint_search.py --config configs/final/joint_ciciiot2025_$m.yaml
  step python experiments/cost_sensitivity.py --config configs/final/cost_ciciiot2025_$m.yaml
done
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
