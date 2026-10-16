# Reproducing each table and figure

`bash scripts/reproduce_paper.sh` runs everything below in order. To reproduce one result, run its experiment(s),
then

```bash
python scripts/latest_runs.py --config configs/paper_artifacts.yaml        # point the config at the newest runs
python scripts/operating_points_all.py --config configs/paper_artifacts.yaml  # only for operating points
python scripts/make_paper_artifacts.py --config configs/paper_artifacts.yaml  # regenerate tables and figures
```

All commands are run from this folder with `PYTHONPATH=src` (or after `pip install -e .`). Experiments need the
prepared data and the detector pools (steps 1 and 2 of `scripts/reproduce_paper.sh`).

| Paper item (section) | Produced by |
| --- | --- |
| Attack categories, datasets after cleaning, class distribution (Sec. 4) | `scripts/prepare_data.py` |
| Detector families and their hyperparameters (Sec. 5) | hand-written; hyperparameters are in `configs/experiments/pool*.yaml` |
| Tuned server-tier detectors (Sec. 5) | `experiments/tune_detectors.py` (results committed in `configs/experiments/pool_strong_*.yaml`) |
| Single detectors: pool table and quality-vs-cost figure (Sec. 6.1, RQ1) | `experiments/build_pool.py` with `configs/experiments/pool.yaml`, `pool_ciciiot2025.yaml`, `pool_strong_*.yaml` |
| Training and inference time (Sec. 6.1) | `experiments/timing.py` |
| Main results, hypervolume vs transmission cost, fronts (Sec. 6.2, RQ2) | `experiments/joint_search.py` with `configs/final/joint_<dataset>.yaml` |
| Tuned two-tier cascade and greedy search (Sec. 6.2) | `experiments/strong_baselines.py` with `configs/final/joint_<dataset>.yaml` |
| Ablation ladder (Sec. 6.3, RQ3) | `experiments/ablation_ladder.py` with `configs/final/ladder_<dataset>.yaml` |
| Design choices: threshold granularity, margin, three objectives (Sec. 6.3) | `joint_search.py` with `joint_`, `joint_margin_` and `joint_3obj_<dataset>.yaml` |
| Confidence by category, escalation diagnostics (Sec. 6.3) | `experiments/escalation_diagnostics.py` |
| Energy profiles, energy budgets, overheads, latency limits (Sec. 6.4, RQ4) | `experiments/cost_sensitivity.py` with `configs/final/cost_<dataset>.yaml` and `energy_baselines_<dataset>.yaml` |
| Routing by category, deployment example and its table (Sec. 6.3, 7) | the energy-profile runs above |
| Operating points, per-category detection, alert precision (Sec. 6.5, RQ5, and 7) | `scripts/operating_points_all.py` after the energy-profile runs |
| Search algorithms and search-seed variance (Sec. 6.6, RQ6) | `experiments/search_study.py`, `experiments/search_seed_variance.py` |
| Later traffic and unseen devices (Sec. 6.6) | `prepare_data.py`, `build_pool.py`, `joint_search.py` and `cost_sensitivity.py` with the `*_temporal` and `*_device` configurations |
| Evasion (Sec. 6.6) | `experiments/evasion.py` |
| Published-protocol replication, comparison with the literature (Sec. 6.7) | `experiments/literature_protocol.py` |
| Edge-IIoTset audit (Sec. 4) | `experiments/edge_leakage_audit.py` |

Section numbers refer to the submitted manuscript; the LaTeX labels written by `scripts/make_paper_artifacts.py`
(for example `tab:operating`) identify each generated table and figure.

## Known differences from the paper

* The search-algorithm study in the paper used an earlier pool of 46 detectors. That pool is no longer built by
  the configurations; `configs/experiments/search_study_final_pool.yaml` runs the same study on the final
  57-detector pool, which gives similar but not identical numbers.
* Detector training uses fixed seeds, but multi-threaded tree learners can still produce tiny numerical
  differences across machines and library versions; `requirements.txt` lists the exact versions used.
