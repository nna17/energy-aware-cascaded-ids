#!/usr/bin/env bash
# After run_strong.sh: re-run the design-choice variants (margin score, three objectives) on the extended pools,
# so that Table design compares runs on identical pools. Progress: results/design_rerun.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/design_rerun.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/design_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }
echo "$(date +%H:%M) waiting for run_strong.sh" >> "$LOG"
until grep -q "ALL DONE" results/strong_pipeline.log 2>/dev/null; do sleep 60; done
for ds in ciciiot2025 ciciot2023; do
  step python experiments/joint_search.py --config configs/final/joint_margin_$ds.yaml
  step python experiments/joint_search.py --config configs/final/joint_3obj_$ds.yaml
done
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
