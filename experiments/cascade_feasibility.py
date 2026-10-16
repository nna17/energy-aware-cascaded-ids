"""Cascade feasibility check.

Does escalating only the uncertain flows from a cheap tier-1 model recover most of a large model's
detection quality (macro-F1, minority recall, benign false positives) at a fraction of its cost?

    python experiments/cascade_feasibility.py --config configs/experiments/cascade_feasibility.yaml

Costs are worst-case tree comparisons per flow; transmission and measured energy come in Phase 1.
"""
from __future__ import annotations

import argparse

import pandas as pd

from eids.cascade import Tier, thresholds_for_rates
from eids.cascade.simulate import cascade_metrics
from eids.data.datamodule import DataModule, load_datamodule
from eids.models.baselines.trees import TreeModel
from eids.utils.config import Config, load_config
from eids.utils.io import make_run_dir, save_json
from eids.utils.logging import get_logger

log = get_logger()


def train(spec: Config, dm: DataModule, seed: int, balanced: bool) -> TreeModel:
    params = {k: v for k, v in spec.items() if k != "kind"}
    model = TreeModel(spec.kind, params, seed).fit(dm.X["train"], dm.y["train"], balanced=balanced)
    log.info(f"trained {model.name}: {model.n_comparisons()} comparisons, {model.size_kb():.1f} KB")
    return model


def tiers_for(models: list[TreeModel], dm: DataModule, split: str) -> list[Tier]:
    return [Tier.from_proba(m.name, m.predict_proba(dm.X[split]), cost=m.n_comparisons()) for m in models]


def evaluate(models: list[TreeModel], rates: list[float], dm: DataModule, val: list[Tier], test: list[Tier]) -> dict:
    """Thresholds from validation confidences; metrics on test. rates[i] = fraction of the flows reaching
    tier i that it escalates (so tier-2 thresholds come from the validation flows tier 1 escalated)."""
    thresholds, reaching = [], None
    for i, rate in enumerate(rates):
        conf = val[i].confidence if reaching is None else val[i].confidence[reaching]
        th = thresholds_for_rates(conf, rate) if len(conf) else -float("inf")
        thresholds.append(th)
        escalated = val[i].confidence < th
        reaching = escalated if reaching is None else reaching & escalated
    m = cascade_metrics(dm.y["test"], test, thresholds, dm.n_classes, dm.labels.minority, dm.labels.benign_index)
    shares = {f"share_t{i + 1}": m.pop(f"share_{t.name}") for i, t in enumerate(test)}
    return {"cascade": " -> ".join(t.name for t in models), "rates": rates, **m, **shares}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/cascade_feasibility.yaml")
    args = parser.parse_args()
    exp = load_config(args.config)
    run_dir = make_run_dir(exp.name)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")

    dm = load_datamodule(load_config(exp.dataset))
    t2, t3 = (train(exp[k], dm, exp.seed, exp.balanced) for k in ("tier2", "tier3"))
    rows = []
    for spec in exp.tier1:
        t1 = train(spec, dm, exp.seed, exp.balanced)
        for models in ([t1], [t2], [t3]):  # each model alone, as references
            rows.append(evaluate(models, [], dm, tiers_for(models, dm, "val"), tiers_for(models, dm, "test")))
        for upper in (t2, t3):  # two tiers
            models = [t1, upper]
            val, test = tiers_for(models, dm, "val"), tiers_for(models, dm, "test")
            rows += [evaluate(models, [r], dm, val, test) for r in exp.tier1_rates if r > 0]
        models = [t1, t2, t3]  # three tiers
        val, test = tiers_for(models, dm, "val"), tiers_for(models, dm, "test")
        rows += [evaluate(models, [r1, r2], dm, val, test)
                 for r1 in exp.tier1_rates if r1 > 0 for r2 in exp.tier2_rates]

    df = pd.DataFrame(rows)
    df["rates"] = df["rates"].astype(str)
    df = df.drop_duplicates(subset=["cascade", "rates"]).reset_index(drop=True)  # tier-2/3 references repeat
    df.to_csv(run_dir / "results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        log.info("Cascade feasibility (test split):\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
