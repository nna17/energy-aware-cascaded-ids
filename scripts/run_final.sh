#!/usr/bin/env bash
# Final pipeline: every experiment of the paper, run in sequence (RAM-limited machine).
# Progress: results/final_pipeline.log. Each step's output: results/final_step_<n>.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
LOG=results/final_pipeline.log
N=0
step() { N=$((N+1)); echo "$(date +%H:%M) START [$N] $*" >> "$LOG"; "$@" > "results/final_step_$N.log" 2>&1; echo "$(date +%H:%M) END   [$N] exit $? : $*" >> "$LOG"; }

# 1. CIC IIoT 2025: add the unrestricted server forest to the existing pools, then every study (seeded search)
step python experiments/build_pool.py --config configs/experiments/pool_ciciiot2025.yaml
for c in joint joint_margin joint_3obj; do step python experiments/joint_search.py --config configs/final/${c}_ciciiot2025.yaml; done
step python experiments/ablation_ladder.py --config configs/final/ladder_ciciiot2025.yaml
step python experiments/cost_sensitivity.py --config configs/final/cost_ciciiot2025.yaml
# 2. CIC-IoT2023: pool v2 (3 seeds, margin, MLPs, unrestricted server forest), then every study
step python experiments/build_pool.py --config configs/experiments/pool.yaml
for c in joint joint_margin joint_3obj; do step python experiments/joint_search.py --config configs/final/${c}_ciciot2023.yaml; done
step python experiments/ablation_ladder.py --config configs/final/ladder_ciciot2023.yaml
step python experiments/cost_sensitivity.py --config configs/final/cost_ciciot2023.yaml
# 3. Audits and replication
step python experiments/edge_leakage_audit.py --config configs/data/edge_iiotset.yaml
step python experiments/literature_protocol.py --data configs/data/ciciiot2025.yaml --sample 500000
step python experiments/literature_protocol.py --data configs/data/ciciot2023.yaml --raw-files 3 --sample 500000
echo "$(date +%H:%M) ALL DONE" >> "$LOG"
