#!/usr/bin/env bash
# Follow-up after run_final.sh: tune strong server-tier detectors (validation only), extend the pools,
# and re-run the main studies. Progress: results/strong_pipeline.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/strong_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/strong_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
echo "$(date +%H:%M) waiting for run_final.sh" >> "$LOG"
until grep -q "ALL DONE" results/final_pipeline.log 2>/dev/null; do sleep 60; done
for ds in ciciiot2025 ciciot2023; do
  step python experiments/tune_detectors.py --data configs/data/$ds.yaml --trials 30 --sample 200000
  step python experiments/build_pool.py --config configs/experiments/pool_strong_$ds.yaml
  step python experiments/joint_search.py --config configs/final/joint_$ds.yaml
  step python experiments/ablation_ladder.py --config configs/final/ladder_$ds.yaml
  step python experiments/cost_sensitivity.py --config configs/final/cost_$ds.yaml
done
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
