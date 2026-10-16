#!/usr/bin/env bash
# Reproduce every experiment of the paper, in order, then regenerate its tables and figures.
# Needs the raw datasets under data/raw/ (see README.md). Roughly two days on a 20-thread laptop CPU with 32 GB RAM.
# Progress: results/reproduce.log; each step's output: results/reproduce_step_<n>.log
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
mkdir -p results
LOG=results/reproduce.log
N=0
step() { N=$((N+1)); echo "$(date '+%F %H:%M') START [$N] $*" >> "$LOG"; "$@" > "results/reproduce_step_$N.log" 2>&1;
         echo "$(date '+%F %H:%M') END   [$N] exit $? : $*" >> "$LOG"; }

# 1. Data: clean, split and cache (random protocol for both datasets; temporal and held-out-device for CIC IIoT 2025)
for d in ciciot2023 ciciiot2025 ciciiot2025_temporal ciciiot2025_device edge_iiotset; do
  step python scripts/prepare_data.py --config configs/data/$d.yaml
done

# 2. Detector pools: 53 grid detectors, then the 4 server-tier detectors. The tuned LightGBM configurations are
#    committed in configs/experiments/pool_strong_*.yaml; to re-tune them, run first:
#      python experiments/tune_detectors.py --data configs/data/<dataset>.yaml --trials 30 --sample 200000
step python experiments/build_pool.py --config configs/experiments/pool.yaml
step python experiments/build_pool.py --config configs/experiments/pool_strong_ciciot2023.yaml
for p in ciciiot2025 ciciiot2025_temporal ciciiot2025_device; do
  step python experiments/build_pool.py --config configs/experiments/pool_$p.yaml
  step python experiments/build_pool.py --config configs/experiments/pool_strong_$p.yaml
done

# 3. Main studies (RQ1-RQ5): joint search and baselines, design variants, ablation ladder, energy profiles
for ds in ciciot2023 ciciiot2025; do
  for c in joint joint_margin joint_3obj ladder; do
    if [ "$c" = ladder ]; then step python experiments/ablation_ladder.py --config configs/final/ladder_$ds.yaml
    else step python experiments/joint_search.py --config configs/final/${c}_$ds.yaml; fi
  done
  step python experiments/cost_sensitivity.py --config configs/final/cost_$ds.yaml
  step python experiments/strong_baselines.py --config configs/final/joint_$ds.yaml
  step python experiments/cost_sensitivity.py --config configs/final/energy_baselines_$ds.yaml
done

# 4. Robustness (RQ6): later traffic and unseen devices, search algorithms and seeds, evasion
for m in temporal device; do
  step python experiments/joint_search.py --config configs/final/joint_ciciiot2025_$m.yaml
  step python experiments/cost_sensitivity.py --config configs/final/cost_ciciiot2025_$m.yaml
done
step python experiments/search_study.py --config configs/experiments/search_study_final_pool.yaml
for ds in ciciot2023 ciciiot2025; do
  step python experiments/search_seed_variance.py --config configs/final/joint_$ds.yaml
done
latest() { echo "results/runs/$1/$(ls -t "results/runs/$1" | head -1)"; }
step python experiments/evasion.py --data configs/data/ciciot2023.yaml --pool-config configs/experiments/pool.yaml \
  --strong configs/experiments/pool_strong_ciciot2023.yaml --fronts "$(latest final_cost_ciciot2023)/fronts.csv" --radio-uj 400
step python experiments/evasion.py --data configs/data/ciciiot2025.yaml \
  --pool-config configs/experiments/pool_ciciiot2025.yaml --strong configs/experiments/pool_strong_ciciiot2025.yaml \
  --fronts "$(latest final_cost_ciciiot2025)/fronts.csv" --radio-uj 600

# 5. Diagnostics, audits, replication of the published protocol, timing
step python experiments/escalation_diagnostics.py --data configs/data/ciciot2023.yaml
step python experiments/edge_leakage_audit.py --config configs/data/edge_iiotset.yaml
step python experiments/literature_protocol.py --data configs/data/ciciiot2025.yaml --sample 500000
step python experiments/literature_protocol.py --data configs/data/ciciot2023.yaml --raw-files 3 --sample 500000
for ds in ciciot2023 ciciiot2025; do
  step python experiments/timing.py --data configs/data/$ds.yaml --extra configs/experiments/pool_strong_$ds.yaml
done

# 6. Point the artifact config at these runs, select operating points, and regenerate tables and figures
step python scripts/latest_runs.py --config configs/paper_artifacts.yaml
step python scripts/operating_points_all.py --config configs/paper_artifacts.yaml
step python scripts/make_paper_artifacts.py --config configs/paper_artifacts.yaml
echo "$(date '+%F %H:%M') ALL DONE" >> "$LOG"
