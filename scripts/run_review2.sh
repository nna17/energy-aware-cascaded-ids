#!/usr/bin/env bash
# Second part of the review experiments (after run_review.sh): stronger baselines and the evasion study.
# Progress: results/review2_pipeline.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/review2_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/review2_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
echo "$(date +%H:%M) waiting for run_review.sh" >> "$LOG"
until grep -q "ALL DONE" results/review_pipeline.log 2>/dev/null; do sleep 60; done
latest() { echo "results/runs/$1/$(ls -t "results/runs/$1" | head -1)"; }
step python experiments/strong_baselines.py --config configs/final/joint_ciciiot2025.yaml
step python experiments/strong_baselines.py --config configs/final/joint_ciciot2023.yaml
step python experiments/evasion.py --data configs/data/ciciiot2025.yaml \
  --pool-config configs/experiments/pool_ciciiot2025.yaml --strong configs/experiments/pool_strong_ciciiot2025.yaml \
  --fronts "$(latest final_cost_ciciiot2025)/fronts.csv" --radio-uj 600
step python experiments/evasion.py --data configs/data/ciciot2023.yaml \
  --pool-config configs/experiments/pool.yaml --strong configs/experiments/pool_strong_ciciot2023.yaml \
  --fronts "results/runs/final_cost_ciciot2023/20261016-022902/fronts.csv" --radio-uj 400
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
