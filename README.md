# Energy-aware cascaded intrusion detection for IoT–edge networks

Code for the paper **"Energy-Aware Multi-Objective Design of Cascaded Intrusion Detection Systems for IoT–Edge
Networks"** by Nabil Abdelkader Nouri, Abdenacer Naouri, Salim Naouri and Ali Soufiane Hassak (under review).

An intrusion detection system for IoT can classify a flow on the device itself, or send it to a gateway or an edge
server where stronger detectors run, at the price of radio energy. This repository optimises such a **cascade**
jointly: which detector runs on each tier, whether each tier is used, and which flows are escalated, with one
confidence threshold per traffic category. The objectives are macro-F1, rare-attack recall, the benign
false-positive rate and the cost per flow, including transmission. Because a cascade is fully determined by each
tier's per-flow predictions, thousands of deployments are evaluated **offline from stored predictions** and searched
with NSGA-II, and the same predictions can be re-scored under any device and radio energy profile without
retraining.

## Contents

| Path | What it contains |
| --- | --- |
| `src/eids/data/` | dataset adapters, cleaning (deduplication, conflicting labels), random / temporal / held-out-device splits |
| `src/eids/models/baselines/` | the detector families: decision trees, random forests, Extra Trees, LightGBM, compact MLPs |
| `src/eids/cascade/` | detector pool, cascade simulator, deployment encoding, search algorithms, operating-point selection |
| `experiments/` | one script per experiment of the paper (see [REPRODUCE.md](REPRODUCE.md)) |
| `scripts/` | data preparation, the full reproduction pipeline, generation of the paper's tables and figures |
| `configs/` | datasets, detector pools, searches and energy profiles; `configs/final/` holds the paper's runs |
| `tests/` | unit tests on synthetic data |

`src/eids/models/elastic`, `src/eids/models/teacher`, `src/eids/distill` and the experiments `e0_pilot`,
`permutation_test` and `cascade_feasibility` belong to an earlier exploration (distillation of a single detector)
and are not used in the paper.

## Installation

Python 3.10 or newer.

```bash
pip install -e ".[dev]"          # or: pip install -r requirements.txt  (exact versions used for the paper)
python -m pytest                 # unit tests on synthetic data, under a minute
```

## Data

The datasets are not redistributed here; download them from their publishers and place them as follows.

| Dataset | Folder | Files used |
| --- | --- | --- |
| CIC-IoT2023 (Neto et al., 2023), MERGED_CSV release | `data/raw/ciciot2023/` | `Merged01.csv` ... (the first 20 of 63 files are read) |
| CIC IIoT 2025 / DataSense (Firouzi et al., 2025) | `data/raw/ciciiot2025/` | `attack_samples_1sec.csv`, `benign_samples_1sec.csv` |
| Edge-IIoTset (Ferrag et al., 2022), only for the leakage audit | `data/raw/edge_iiotset/` | `DNN-EdgeIIoT-dataset.csv` |

CIC-IoT2023 and CIC IIoT 2025 are published by the Canadian Institute for Cybersecurity
(<https://www.unb.ca/cic/datasets/>). Then prepare the cached, cleaned splits:

```bash
python scripts/prepare_data.py --config configs/data/ciciot2023.yaml
python scripts/prepare_data.py --config configs/data/ciciiot2025.yaml
```

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh
```

runs every experiment in order (about two days on a 20-thread laptop CPU with 32 GB of RAM), points
`configs/paper_artifacts.yaml` at the new runs and regenerates every table and figure into `../paper/`.
[REPRODUCE.md](REPRODUCE.md) lists which experiment produces each table and figure, and how to run a single one.

Each run writes to `results/runs/<experiment>/<timestamp>/` (`config.json`, `summary.csv`, `fronts.csv`, logs).
Pools of stored per-flow predictions are written to `results/pools_*/<dataset>/seed<k>/`.

## Citation

See [CITATION.cff](CITATION.cff). The reference will be updated when the paper is published.

## Licence

MIT, see [LICENSE](LICENSE). The datasets keep their own licences.
