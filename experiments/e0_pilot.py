"""E0 pilot / decision gate 1.

Do small elastic sub-models (distilled once) keep minority-attack recall better than
tree ensembles of similar size and fixed students trained one by one?

    python experiments/e0_pilot.py --config configs/experiments/e0_pilot.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from eids.data.datamodule import DataModule, load_datamodule
from eids.eval.evaluator import evaluate, evaluate_subnet, metrics_from_predictions
from eids.models.baselines.fixed_student import build_fixed_student
from eids.models.baselines.trees import tree_grid
from eids.models.elastic import ElasticMLP, SearchSpace, macs, param_count
from eids.models.elastic.subnet import size_kb
from eids.models.teacher import TeacherEnsemble, build_teacher
from eids.training.classifier import train_classifier
from eids.training.supernet import train_supernet
from eids.utils.config import Config, load_config
from eids.utils.io import load_json, make_run_dir, save_json
from eids.utils.logging import get_logger
from eids.utils.seed import resolve_device, set_seed

log = get_logger()
HEADLINE = ["macro_f1", "minority_recall", "accuracy", "false_positive_rate", "detection_rate"]


class Recorder:
    def __init__(self, dataset: str, seed: int):
        self.dataset, self.seed = dataset, seed
        self.rows: list[dict] = []
        self.per_class: list[dict] = []

    def add(self, method: str, model: str, metrics: dict, n_params: int | None = None,
            size_kb_value: float | None = None, size_kb_int8: float | None = None, shape: str | None = None,
            ops: int | None = None):
        # ops = inference cost per prediction: multiply-adds for MLPs, worst-case comparisons for trees.
        self.rows.append({
            "dataset": self.dataset, "seed": self.seed, "method": method, "model": model, "shape": shape,
            "n_params": n_params, "size_kb": size_kb_value, "size_kb_int8": size_kb_int8, "ops": ops,
            **{k: metrics.get(k) for k in HEADLINE},
        })
        self.per_class.append({"dataset": self.dataset, "seed": self.seed, "method": method, "model": model,
                               "per_class_recall": metrics["per_class_recall"], "support": metrics["support"]})
        log.info(f"[{self.dataset}] {method:<14} {model:<28} macroF1={metrics['macro_f1']:.4f} "
                 f"minRec={metrics['minority_recall']:.4f} FPR={metrics.get('false_positive_rate', float('nan')):.4f} "
                 f"ops={ops}")


def run_trees(exp: Config, dm: DataModule, rec: Recorder, seed: int) -> None:
    t = exp.trees
    for tree in tree_grid(t.kinds, t.n_estimators, t.max_leaves, seed):
        tree.fit(dm.X["train"], dm.y["train"], balanced=t.get("class_weighting") == "balanced")
        metrics = metrics_from_predictions(dm, "test", tree.predict(dm.X["test"]))
        rec.add("tree", tree.name, metrics, size_kb_value=tree.size_kb(), size_kb_int8=tree.size_kb(),
                ops=tree.n_comparisons())


def run_teacher(exp: Config, dm: DataModule, rec: Recorder, device: torch.device, seed: int) -> torch.nn.Module:
    train_cfg = load_config(exp.teacher.train)
    members = []
    for path in exp.teacher.members:
        model_cfg = load_config(path)
        model = build_teacher(model_cfg, dm.n_features, dm.n_classes)
        train_classifier(model, dm, train_cfg, device, seed, name=f"teacher_{model_cfg.kind}")
        rec.add("teacher", model_cfg.kind, evaluate(model, dm, "test", device),
                n_params=sum(p.numel() for p in model.parameters()))
        members.append(model)
    if len(members) == 1:
        return members[0]
    ensemble = TeacherEnsemble(members)
    rec.add("teacher", "ensemble", evaluate(ensemble, dm, "test", device))
    return ensemble


def run_supernet(exp: Config, dm: DataModule, rec: Recorder, device: torch.device, seed: int,
                 teacher: torch.nn.Module | None, method: str) -> ElasticMLP:
    model_cfg, train_cfg = load_config(exp.supernet.model), load_config(exp.supernet.train)
    if teacher is None:  # baseline B4: no teacher and no in-place distillation
        train_cfg.update(use_kd=False, inplace_distillation=False)
    space = SearchSpace.from_config(model_cfg, dm.n_features)
    supernet = ElasticMLP(dm.n_features, dm.n_classes, model_cfg.max_widths, model_cfg.get("dropout", 0.0))
    set_seed(seed)
    train_supernet(supernet, space, dm, train_cfg, device, seed, teacher=teacher, name=method)

    for s in exp.subnets:
        sub = space.uniform(s.depth, s.width_ratio, s.feature_ratio)
        metrics = evaluate_subnet(supernet, sub, dm, "test", device, train_cfg.get("bn_recalib_batches", 50))
        n = param_count(sub, dm.n_classes)
        rec.add(method, f"{s.name} {sub.widths} f{sub.n_features}", metrics, n_params=n,
                size_kb_value=size_kb(n, 32), size_kb_int8=size_kb(n, 8), shape=s.name,
                ops=macs(sub, dm.n_classes))
    return supernet


def run_fixed_students(exp: Config, dm: DataModule, rec: Recorder, device: torch.device, seed: int,
                       teacher: torch.nn.Module) -> None:
    model_cfg, sn_cfg = load_config(exp.supernet.model), load_config(exp.supernet.train)
    space = SearchSpace.from_config(model_cfg, dm.n_features)
    train_cfg = load_config(exp.teacher.train)
    train_cfg.update(epochs=sn_cfg.epochs, lr=sn_cfg.lr, class_weighting=sn_cfg.class_weighting)
    shapes = {s.name: s for s in exp.subnets}
    for name in exp.fixed_students:
        s = shapes[name]
        sub = space.uniform(s.depth, s.width_ratio, s.feature_ratio)
        set_seed(seed)
        student = build_fixed_student(sub, dm.n_classes)
        train_classifier(student, dm, train_cfg, device, seed, teacher=teacher, kd_cfg=sn_cfg,
                         name=f"fixed_{name}")
        n = param_count(sub, dm.n_classes)
        rec.add("fixed_kd", f"{name} {sub.widths} f{sub.n_features}", evaluate(student, dm, "test", device),
                n_params=n, size_kb_value=size_kb(n, 32), size_kb_int8=size_kb(n, 8), shape=name,
                ops=macs(sub, dm.n_classes))


GATE_METRICS = {"minrec": "minority_recall", "f1": "macro_f1", "fpr": "false_positive_rate"}


def _best_tree(trees: pd.DataFrame, prefix: str) -> dict:
    """Best tree by minority recall among those under a budget (averaged over seeds)."""
    if not len(trees):
        return {f"{prefix}": "none fits", **{f"{prefix}_{k}": None for k in GATE_METRICS}}
    agg = trees.groupby("model")[list(GATE_METRICS.values())].mean().sort_values("minority_recall")
    best = agg.iloc[-1]
    return {f"{prefix}": best.name, **{f"{prefix}_{k}": best[v] for k, v in GATE_METRICS.items()}}


def decision_gate(results: pd.DataFrame) -> pd.DataFrame:
    """Per sub-model shape: elastic KD vs no-KD vs fixed KD student vs the best tree that fits
    (a) the same INT8 size and (b) the same inference cost (tree comparisons <= MLP multiply-adds)."""
    rows = []
    for (dataset, shape), g in results[results["shape"].notna()].groupby(["dataset", "shape"]):
        row = {"dataset": dataset, "shape": shape}
        for method in ("elastic_kd", "elastic_nokd", "fixed_kd"):
            m = g[g["method"] == method]
            for k, v in GATE_METRICS.items():
                row[f"{method}_{k}"] = m[v].mean() if len(m) else None
        size_budget, ops_budget = g["size_kb_int8"].iloc[0], g["ops"].iloc[0]
        trees = results[(results["dataset"] == dataset) & (results["method"] == "tree")]
        row.update(budget_kb_int8=size_budget, budget_ops=ops_budget)
        row.update(_best_tree(trees[trees["size_kb"] <= size_budget], "tree_by_size"))
        row.update(_best_tree(trees[trees["ops"] <= ops_budget], "tree_by_ops"))
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(run_dir: Path, rows: list[dict], per_class: list[dict]) -> None:
    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "results.csv", index=False)
    save_json(per_class, run_dir / "per_class.json")
    if not results["shape"].notna().any() or not (results["method"] == "tree").any():
        log.info("results saved; decision gate needs both tree and network results (see --merge)")
        return
    gate = decision_gate(results)
    gate.to_csv(run_dir / "decision_gate.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        log.info("Decision gate 1 (test split):\n" + gate.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/e0_pilot.yaml")
    parser.add_argument("--datasets", nargs="*", help="override the dataset config list")
    # Trees use the CPU and the networks use the GPU, so they can run as two parallel processes:
    #   --parts trees   and   --parts nn   then   --merge <run_dir_a> <run_dir_b>
    parser.add_argument("--parts", nargs="+", choices=["trees", "nn"], default=["trees", "nn"])
    parser.add_argument("--merge", nargs="+", metavar="RUN_DIR", help="combine finished runs into one decision gate")
    args = parser.parse_args()

    exp = load_config(args.config)
    if args.merge:
        run_dir = make_run_dir(f"{exp.name}_merged")
        get_logger(log_file=run_dir / "log.txt")
        rows, per_class = [], []
        for d in map(Path, args.merge):
            rows += pd.read_csv(d / "results.csv").to_dict("records")
            per_class += load_json(d / "per_class.json")
        save_json({"merged_from": args.merge}, run_dir / "config.json")
        write_outputs(run_dir, rows, per_class)
        return

    device = resolve_device(exp.get("device", "auto"))
    suffix = "" if set(args.parts) == {"trees", "nn"} else "_" + "_".join(args.parts)
    run_dir = make_run_dir(exp.name + suffix)
    get_logger(log_file=run_dir / "log.txt")
    save_json(exp, run_dir / "config.json")
    log.info(f"run dir {run_dir} | device {device} | parts {args.parts}")

    recorders = []

    def checkpoint() -> None:  # save after every step, so a crash keeps finished results
        write_outputs(run_dir, [r for rec in recorders for r in rec.rows],
                      [p for rec in recorders for p in rec.per_class])

    for data_path in args.datasets or exp.datasets:
        dm = load_datamodule(load_config(data_path))
        for seed in exp.seeds:
            set_seed(seed)
            rec = Recorder(dm.name, seed)
            recorders.append(rec)
            if "trees" in args.parts:
                run_trees(exp, dm, rec, seed)
                checkpoint()
            if "nn" in args.parts:
                teacher = run_teacher(exp, dm, rec, device, seed)
                supernet = run_supernet(exp, dm, rec, device, seed, teacher, "elastic_kd")
                torch.save({"teacher": teacher.state_dict(), "supernet": supernet.state_dict()},
                           run_dir / f"{dm.name}_seed{seed}_models.pt")
                checkpoint()
                if exp.supernet.get("ablation_no_kd", True):
                    run_supernet(exp, dm, rec, device, seed, None, "elastic_nokd")
                    checkpoint()
                run_fixed_students(exp, dm, rec, device, seed, teacher)
                checkpoint()


if __name__ == "__main__":
    main()
