"""Generate every result table (LaTeX) and figure (PDF) of the paper from experiment outputs.

Which run feeds which artifact is listed in configs/paper_artifacts.yaml; after new runs, update that file
and re-run. Generated files carry a header saying so: edit this script, not the outputs.

    python scripts/make_paper_artifacts.py --config configs/paper_artifacts.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paper_artifact_parts import (ALGOS, FAMILIES, GRID, INK, INK_2, class_distribution_figure,
                                  confidence_by_category_figure, fmt_int, fmt_size, plt, search_curve_figure,
                                  search_study_table, style, write)

from eids.cascade.pool import load_pool
from eids.eval.metrics import fast_metrics
from eids.utils.config import load_config, resolve_path
from eids.utils.io import load_json

METHODS = {  # summary key -> (label, colour, marker); validated palette order
    "single_model": ("Single detector", "#2a78d6", "o"),
    "fixed_two_tier": ("Fixed two-tier (C2T-style)", "#eb6834", "s"),
    "random_category": ("Random search", "#eda100", "^"),
    "nsga2_category": ("Joint (NSGA-II)", "#1baf7a", "D"),
}
POOL_ROWS = [
    ("decision_tree_l16", "Decision tree, 16 leaves"),
    ("decision_tree_l256_f20", "Decision tree, 256 leaves, 20 features"),
    ("decision_tree_l1024", "Decision tree, 1,024 leaves"),
    ("mlp_32_f20", "MLP, 32 hidden units, 20 features"),
    ("extra_trees_n10_l512", r"Extra Trees, $10\times512$"),
    ("random_forest_n10_l512", r"Random forest, $10\times512$"),
    ("random_forest_n50_lNone", "Random forest, 50 unrestricted trees"),
    ("lightgbm_n50_l32", r"LightGBM, $50\times32$"),
    ("lightgbm_n200_l32", r"LightGBM, $200\times32$"),
    ("random_forest_n100_lNone_unw", "Random forest, 100 unrestricted trees, unweighted"),
    (r"lightgbm_n\d+_l\d+_tuned_f1", "LightGBM, tuned for macro-F1"),
]
LADDER = [
    ("device_only", "Device only"),
    ("device_gateway_fixed", "Device $\\rightarrow$ gateway, fixed rule"),
    ("gateway_server_c2t", "Gateway $\\rightarrow$ server, fixed rule (C2T-style)"),
    ("models_only", "Detectors and tiers optimised, thresholds fixed"),
    ("thresholds_only", "Thresholds optimised, detectors fixed"),
    ("joint", "Joint optimisation"),
]
PROFILES = [
    ("system_wifi", "System energy, Wi-Fi"), ("system_ble", "System energy, BLE"),
    ("system_802154", "System energy, IEEE 802.15.4"), ("device_wifi", "Device energy, Wi-Fi"),
    ("device_ble", "Device energy, BLE"), ("device_802154", "Device energy, IEEE 802.15.4"),
    ("system_ble_compute_x0.1", "BLE, computation $\\times 0.1$"), ("system_ble_compute_x10", "BLE, computation $\\times 10$"),
    ("system_ble_radio_x0.1", "BLE, radio $\\times 0.1$"), ("system_ble_radio_x10", "BLE, radio $\\times 10$"),
]


def pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}" if not np.isfinite(std) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def pct(x: float) -> str:
    return f"{x:+.0f}\\%" if abs(x) >= 10 else f"{x:+.1f}\\%"


def table(path: Path, caption: str, label: str, colspec: str, head: str, body: list[str], wide: bool = False) -> None:
    # Wide tables span both columns of the journal layout, narrow ones sit in one column; either is shrunk to fit
    # if needed, never enlarged.
    width, env, font = ("\\textwidth", "table*", "\\small") if wide else ("\\linewidth", "table", "\\footnotesize")
    open_box, close_box = f"\\begin{{adjustbox}}{{max width={width}}}\n", "\n\\end{adjustbox}"
    write(path, f"""\\begin{{{env}}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{font}
{open_box}\\begin{{tabular}}{{{colspec}}}
\\toprule
{head}
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}{close_box}
\\end{{{env}}}
""")


def run_csv(run: str, name: str) -> pd.DataFrame:
    return pd.read_csv(resolve_path(run) / name)


# ---------------------------------------------------------------- datasets

def datasets_table(cfg, out: Path) -> None:
    metas = {k: load_json(resolve_path(d.processed) / "meta.json") for k, d in cfg.datasets.items()}
    keys = list(metas)

    def cell(fn):
        return " & ".join(fn(metas[k], k) for k in keys)

    def dup(m, k):
        within, read = m["loading"]["duplicates_within_chunks"], m["loading"]["rows_read"]
        return f"{fmt_int(within + m['cleaning']['exact_duplicates'])} ({100 * within / read:.1f}\\%)"

    def benign_share(m, k):
        c = m["class_counts"]["train"]
        return f"{100 * c[m['classes'][m['benign_index']]] / sum(c.values()):.0f}\\%"

    rows = [
        ("Records read", lambda m, k: fmt_int(m["loading"]["rows_read"])),
        ("Exact duplicates (share)", dup),
        ("After per-class sampling", lambda m, k: fmt_int(m["loading"]["rows_kept"]) if m["loading"]["rows_kept"] < m["loading"]["rows_read"] - m["loading"]["duplicates_within_chunks"] else "no sampling"),
        ("Contradictory labels", lambda m, k: fmt_int(m["cleaning"]["conflicting_rows"])),
        ("Final records", lambda m, k: f"\\textbf{{{fmt_int(m['cleaning']['rows_out'])}}}"),
        ("Train / val. / test", lambda m, k: "/".join(fmt_int(sum(m["class_counts"][s].values())) for s in ("train", "val", "test"))),
        ("Classes (minority)", lambda m, k: f"{len(m['classes'])} ({len(m['minority'])})"),
        ("Flow features", lambda m, k: str(len(m["feature_names"]))),
        ("Benign share (training)", benign_share),
    ]
    head = "Quantity & " + " & ".join(cfg.datasets[k].label for k in keys) + " \\\\"
    body = [f"{name} & {cell(fn)} \\\\" for name, fn in rows]
    table(out, "Datasets after cleaning. Duplicates are exact copies of a record (all features and the label); "
               "contradictory labels are records whose features are identical but whose labels differ.",
          "tab:datasets", "@{}l" + "r" * len(keys) + "@{}", head, body)


# ---------------------------------------------------------------- detector pool

def pool_metrics(pool) -> pd.DataFrame:
    y = pool.y["test"].astype(np.int64)
    rows = []
    for i, name in enumerate(pool.names):
        f1, mr, fpr = fast_metrics(y, pool.pred["test"][i].astype(np.int64), pool.n_classes, pool.minority, pool.benign_index)
        family = next((f for f in FAMILIES if name.startswith(f)), "mlp" if name.startswith("mlp") else "other")
        rows.append({"name": name, "family": family, "cost": pool.cost[i], "size_kb": pool.size_kb[i],
                     "features": int(pool.n_features[i]), "macro_f1": f1, "minority_recall": mr, "fpr": fpr})
    return pd.DataFrame(rows)


def pool_table(cfg, pools: dict[str, pd.DataFrame], out: Path) -> None:
    keys = list(pools)
    head = ("Detector & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(keys))) + "\n"
            + " & " + " & ".join("Ops & Macro-F1 & Min.\\ rec. & FPR" for _ in keys) + " \\\\")
    body = []
    for name, label in POOL_ROWS:
        cells = []
        for k in keys:
            r = pools[k][pools[k].name.str.fullmatch(name)].iloc[0]  # names may be patterns (tuned sizes differ)
            cells.append(f"{fmt_int(r.cost)} & {r.macro_f1:.3f} & {r.minority_recall:.3f} & {r.fpr:.3f}")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table(out, "Selected detectors of the pool used alone (test split, seed 0). Ops: worst-case comparisons (trees) or "
               "multiply-accumulates (MLP) per flow. Detectors use balanced class weights unless marked unweighted; the "
               "tuned LightGBM's hyperparameters and class weighting were tuned for validation macro-F1.",
          "tab:pool", "@{}l" + "rrrr" * len(keys) + "@{}", head, body, wide=True)


def quality_cost_figure(cfg, pools: dict[str, pd.DataFrame], out: Path) -> None:
    keys = list(pools)
    fig, axes = plt.subplots(2, len(keys), figsize=(6.6, 4.6), sharex="col")
    fams = dict(FAMILIES, mlp=("MLP", "#e87ba4", "v"))
    for j, k in enumerate(keys):
        df = pools[k]
        for i, (col, ylabel) in enumerate([("macro_f1", "Macro-F1"), ("fpr", "Benign FPR")]):
            ax = axes[i, j]
            for fam, (name, color, marker) in fams.items():
                d = df[df.family == fam]
                if len(d):
                    ax.scatter(d.cost, d[col], s=18, color=color, marker=marker, label=name,
                               edgecolors="white", linewidths=0.5, zorder=3)
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(cfg.datasets[k].label, loc="left", color=INK)
            if i == 1:
                ax.set_xlabel("Operations per flow (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- main comparison

def main_results(cfg) -> dict[str, pd.DataFrame]:
    out = {}
    for k, d in cfg.datasets.items():
        s = run_csv(d.joint, "summary.csv")
        out[k] = s[s.study == "tx_sweep"]
    return out


def main_table(cfg, sweeps: dict[str, pd.DataFrame], out: Path) -> None:
    head = ("Dataset & $t$ & " + " & ".join(METHODS[m][0] for m in METHODS)
            + " & vs.\\ fixed & vs.\\ random & vs.\\ single \\\\")
    body = []
    for k, s in sweeps.items():
        g = s.groupby(["tx", "method"]).hv_log_cost.agg(["mean", "std"])
        txs = sorted(s.tx.unique())
        for i, tx in enumerate(txs):
            cells = [pm(g.loc[(tx, m), "mean"], g.loc[(tx, m), "std"]) for m in METHODS]
            joint = g.loc[(tx, "nsga2_category"), "mean"]
            gains = [pct(100 * (joint / g.loc[(tx, m), "mean"] - 1)) for m in ("fixed_two_tier", "random_category", "single_model")]
            name = f"\\multirow{{{len(txs)}}}{{*}}{{{cfg.datasets[k].label}}}" if i == 0 else ""
            body.append(f"{name} & {fmt_int(tx)} & " + " & ".join(cells[:-1]) + f" & \\textbf{{{cells[-1]}}} & " + " & ".join(gains) + " \\\\")
        body.append("\\midrule")
    table(out, "Hypervolume (log cost, test split, mean $\\pm$ std over three seeds) against the transmission cost $t$ per "
               "hop in comparison-equivalents, and the gain of joint optimisation over each alternative.",
          "tab:main", "@{}lr" + "c" * len(METHODS) + "rrr@{}", head, body[:-1], wide=True)


def hv_tx_figure(cfg, sweeps: dict[str, pd.DataFrame], out: Path) -> None:
    keys = list(sweeps)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.45, 2.6), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), keys):
        s = sweeps[k]
        txs = sorted(s.tx.unique())
        for m, (label, color, marker) in METHODS.items():
            g = s[s.method == m].groupby("tx").hv_log_cost.agg(["mean", "std"]).reindex(txs)
            ax.errorbar(range(len(txs)), g["mean"], yerr=g["std"], color=color, marker=marker, linewidth=1.6,
                        markersize=5, capsize=2, markeredgecolor="white", markeredgewidth=0.5, label=label)
        ax.set_xticks(range(len(txs)), [f"{t:g}" if t < 1000 else f"{t // 1000:g}k" for t in txs], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Transmission cost $t$", fontsize=8)
        ax.set_title(cfg.datasets[k].label, loc="left", color=INK)
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].set_ylabel("Hypervolume (log cost)", fontsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.8), w_pad=0.6)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fronts_figure(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.8))
    for ax, k in zip(np.atleast_1d(axes), keys):
        f = run_csv(cfg.datasets[k].joint, "fronts.csv")
        f = f[(f.study == "main") & (f.seed == 0)]
        for m, (label, color, marker) in METHODS.items():
            d = f[f.method == m]
            if d.empty:
                continue
            cost = np.maximum(d.mean_cost.to_numpy(), 1.0)
            order = np.argsort(cost)
            c, q = cost[order], np.maximum.accumulate(d.macro_f1.to_numpy()[order])
            keep = np.r_[True, np.diff(q) > 0]
            ax.scatter(cost, d.macro_f1, s=6, color=color, alpha=0.25, linewidths=0)
            ax.step(c[keep], q[keep], where="post", color=color, linewidth=1.5)
            ax.plot(c[keep], q[keep], linestyle="none", marker=marker, markersize=3.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.4, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Mean operations per flow (log scale)")
        ax.set_title(cfg.datasets[k].label, loc="left", color=INK)
    np.atleast_1d(axes)[0].set_ylabel("Macro-F1")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- design choices, ladder, cost

def design_table(cfg, out: Path) -> None:
    rows = [("Thresholds: one global value", "joint", "nsga2_global"),
            ("Thresholds: one per attack category (default)", "joint", "nsga2_category"),
            ("Thresholds: one per class", "joint", "nsga2_class"),
            ("Escalation score: margin instead of confidence", "margin", "nsga2_category"),
            ("Objectives: without minority recall", "three_objectives", "nsga2_category"),
            ("Random search, per category (same budget)", "joint", "random_category")]
    keys = list(cfg.datasets)
    body = []
    for label, run, method in rows:
        cells = []
        for k in keys:
            s = run_csv(cfg.datasets[k][run], "summary.csv")
            v = s[(s.study == "main") & (s.method == method)].hv_log_cost
            cells.append(pm(v.mean(), v.std()))
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    head = "Variant (no transmission cost) & " + " & ".join(cfg.datasets[k].label for k in keys) + " \\\\"
    table(out, "Design choices of the joint search: hypervolume (log cost, test split, mean $\\pm$ std over three seeds) "
               "without transmission cost. The minority-recall ablation is scored on all four objectives.",
          "tab:design", "@{}l" + "c" * len(keys) + "@{}", head, body)


def ladder_table(cfg, out: Path) -> None:
    body = []
    keys = list(cfg.datasets)
    txs = None
    for k in keys:
        s = run_csv(cfg.datasets[k].ladder, "summary.csv")
        g = s.groupby(["variant", "tx"]).hv_log_cost.mean()
        txs = sorted(s.tx.unique())
        for i, (v, label) in enumerate(LADDER):
            name = f"\\multirow{{{len(LADDER)}}}{{*}}{{{cfg.datasets[k].label}}}" if i == 0 else ""
            cells = [f"{g.loc[(v, t)]:.3f}" for t in txs]
            if v == "joint":
                cells = [f"\\textbf{{{c}}}" for c in cells]
            body.append(f"{name} & {label} & " + " & ".join(cells) + " \\\\")
        body.append("\\midrule")
    head = "Dataset & Variant (same tiers and detector pool) & " + " & ".join(f"$t={fmt_int(t)}$" for t in txs) + " \\\\"
    table(out, "Ablation ladder: hypervolume (log cost, test split, mean over three seeds; standard deviations are at most "
               "0.005) when only part of the deployment is optimised, against the transmission cost $t$ per hop.",
          "tab:ladder", "@{}ll" + "c" * len(txs) + "@{}", head, body[:-1], wide=True)


def cost_table(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    stats = {}
    for k in keys:
        s = run_csv(cfg.datasets[k].cost, "summary.csv")
        stats[k] = s.groupby(["profile", "method"]).hv_log_cost.agg(["mean", "std"])
    head = ("Energy profile & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(keys))) + "\n"
            + " & " + " & ".join("C2T & Best single & Joint & Gain" for _ in keys) + " \\\\")
    body = []
    for p, label in PROFILES:
        cells = []
        for k in keys:
            g = stats[k].loc[p]
            alt = g.drop(index="joint_nsga2")["mean"].max()
            cells.append(f"{g.loc['gateway_server_c2t', 'mean']:.3f} & {g.loc['single_model', 'mean']:.3f} & "
                         f"\\textbf{{{g.loc['joint_nsga2', 'mean']:.3f}}} & {pct(100 * (g.loc['joint_nsga2', 'mean'] / alt - 1))}")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table(out, "Cost sensitivity under model-based energy profiles (Table~\\ref{tab:costs}): hypervolume (log energy, test "
               "split, mean over three seeds). Gain: joint optimisation over the best other method (single detector, "
               "device-only, device-first fixed, or C2T-style).", "tab:cost", "@{}l" + "rrrr" * len(keys) + "@{}", head, body, wide=True)


def cost_figure(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    shown = {"gateway_server_c2t": METHODS["fixed_two_tier"], "single_model": METHODS["single_model"],
             "joint_nsga2": METHODS["nsga2_category"]}
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 3.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        g = run_csv(cfg.datasets[k].cost, "summary.csv").groupby(["profile", "method"]).hv_log_cost.mean()
        ys = np.arange(len(PROFILES))[::-1]
        for m, (label, color, marker) in shown.items():
            ax.scatter([g.loc[(p, m)] for p, _ in PROFILES], ys, color=color, marker=marker, s=26, label=label,
                       edgecolors="white", linewidths=0.5, zorder=3)
        for y, (p, _) in zip(ys, PROFILES):
            ax.plot([g.loc[(p, "gateway_server_c2t")], g.loc[(p, "joint_nsga2")]], [y, y], color=GRID, linewidth=1, zorder=1)
        ax.set_title(cfg.datasets[k].label, loc="left", color=INK)
        ax.set_xlabel("Hypervolume (log energy)")
    labels = [lab.replace("$\\times 0.1$", "×0.1").replace("$\\times 10$", "×10") for _, lab in PROFILES]
    np.atleast_1d(axes)[0].set_yticks(np.arange(len(PROFILES))[::-1], labels, fontsize=7)
    handles, hl = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, hl, loc="upper center", ncol=3, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- quality under an energy budget

BUDGET_PROFILES = [("system_ble", "System energy, BLE"), ("device_ble", "Device battery, BLE")]
BUDGET_METHODS = {"device_only": ("Device only", "#e87ba4", "v"),
                  "single_model": METHODS["single_model"],
                  "gateway_server_c2t": METHODS["fixed_two_tier"],
                  "tuned_two_tier": ("Tuned two-tier", "#eda100", "^"),
                  "greedy": ("Greedy search", "#6250d6", "P"),
                  "joint_nsga2": METHODS["nsga2_category"]}
BUDGETS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TABLE_BUDGETS = [0.01, 0.03, 0.1, 0.3]


def energy_fronts(d) -> pd.DataFrame:
    """Fronts of a dataset's energy-profile run, with the stronger baselines of its energy_baselines run if any."""
    f = run_csv(d.cost, "fronts.csv")
    if d.get("energy_baselines"):
        f = pd.concat([f, run_csv(d.energy_baselines, "fronts.csv")], ignore_index=True)
    return f


def budget_curves(d, profile: str) -> pd.DataFrame:
    """Best test macro-F1 per method when deployments may use at most a share of the reference detector's energy.

    The reference is the single detector with the best validation macro-F1 (send every flow to it); deployments are
    selected on validation and reported on test. Rows: seed, budget, method, f1, fpr, ref_f1, floor_f1."""
    f = energy_fronts(d)
    rows = []
    for seed, g in f[f.profile == profile].groupby("seed"):
        singles = g[g.method == "single_model"]
        ref = singles.loc[singles.val_macro_f1.idxmax()]
        dev = g[g.method == "device_only"]
        floor = dev.loc[dev.val_macro_f1.idxmax()].macro_f1
        for b in BUDGETS:
            for m in BUDGET_METHODS:
                ok = g[(g.method == m) & (g.val_mean_cost <= b * ref.val_mean_cost)]
                if ok.empty:
                    continue
                d = ok.loc[ok.val_macro_f1.idxmax()]
                rows.append(dict(seed=seed, budget=b, method=m, f1=d.macro_f1, fpr=d.fpr, ref_f1=ref.macro_f1,
                                 floor_f1=floor, ref_energy=ref.mean_cost))
    return pd.DataFrame(rows)


def budget_figure(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    fig, axes = plt.subplots(len(BUDGET_PROFILES), len(keys), figsize=(6.6, 5.2), sharex=True, squeeze=False)
    for j, k in enumerate(keys):
        for i, (p, plabel) in enumerate(BUDGET_PROFILES):
            ax = axes[i, j]
            r = budget_curves(cfg.datasets[k], p)
            ax.axhline(r.ref_f1.mean(), color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
            ax.annotate("best detector, every flow sent", (0.02, r.ref_f1.mean()), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK_2)
            for m, (label, color, marker) in BUDGET_METHODS.items():
                g = r[r.method == m].groupby("budget").f1
                n = g.size()
                g = g.mean()[n == r.seed.nunique()]  # only budgets feasible on every seed
                device = m == "device_only"  # drawn dashed on top: at small budgets the best single detector is this one
                ax.plot(100 * g.index, g.values, color=color, marker=marker, markersize=4.5, linewidth=1.6,
                        linestyle=(0, (4, 2)) if device else "-", label=label, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=4 if device else 3)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}%"))
            ax.set_title(f"{cfg.datasets[k].label}: {plabel}", loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("Macro-F1 (test)")
    handles, hl = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, hl, loc="upper center", ncol=4, frameon=False, fontsize=7.5)
    fig.supxlabel("Energy budget (share of the energy of sending every flow to the best detector)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def budget_table(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    shown = ["device_only", "single_model", "gateway_server_c2t", "tuned_two_tier", "greedy", "joint_nsga2"]
    w = len(shown) + 1
    head = ("Budget & " + " & ".join(f"\\multicolumn{{{w}}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + w * i}-{1 + w * (i + 1)}}}" for i in range(len(keys))) + "\n"
            + " & " + " & ".join("Device & Single & C2T & Tuned & Greedy & Joint & Gap" for _ in keys) + " \\\\")
    body = []
    for p, plabel in BUDGET_PROFILES:
        curves = {k: budget_curves(cfg.datasets[k], p) for k in keys}
        body.append(f"\\multicolumn{{{1 + w * len(keys)}}}{{@{{}}l}}{{\\textit{{{plabel}}}}} \\\\")
        for b in TABLE_BUDGETS:
            cells = []
            for k in keys:
                r = curves[k][curves[k].budget == b]
                nseeds = curves[k].seed.nunique()
                for m in shown:
                    g = r[r.method == m]
                    v = f"{g.f1.mean():.3f}" if len(g) == nseeds else "--"
                    cells.append(f"\\textbf{{{v}}}" if m == "joint_nsga2" and v != "--" else v)
                j = r[r.method == "joint_nsga2"]
                gap = ((j.f1 - j.floor_f1) / (j.ref_f1 - j.floor_f1)).mean() if len(j) == nseeds else np.nan
                cells.append("--" if np.isnan(gap) else f"{100 * gap:.0f}\\%")
            body.append(f"{100 * b:g}\\% & " + " & ".join(cells) + " \\\\")
    refs = ", ".join(f"{cfg.datasets[k].label} {budget_curves(cfg.datasets[k], BUDGET_PROFILES[0][0]).ref_f1.mean():.3f}"
                     for k in keys)
    table(out, "Best macro-F1 (test, mean over three seeds) under an energy budget, as a share of the energy of sending "
               "every flow to the best single detector (macro-F1: " + refs + "). Deployments are selected on validation. "
               "Device: device-only detection; Single: best single detector on any tier; C2T: fixed gateway--server "
               "design; Tuned: two-tier cascade with a validation-tuned global threshold; Greedy: greedy search with "
               "the budget of joint optimisation; --: no deployment of that method fits the budget. Gap: share of the macro-F1 gap between "
               "device-only detection and the best detector that joint optimisation recovers.",
          "tab:budget", "@{}l" + "r" * w * len(keys) + "@{}", head, body, wide=True)


SCENARIO_PROFILE = "device_ble"
FLOWS_PER_DAY = 2000
COIN_CELL_J = 2430  # CR2032: about 225 mAh at 3 V


def scenario_rows(run: str, profile: str) -> pd.DataFrame:
    """Send-all, device-only and joint deployments at fixed budgets (selected on val; test metrics, mean over seeds)."""
    f = run_csv(run, "fronts.csv")
    rows = []
    for seed, g in f[f.profile == profile].groupby("seed"):
        s = g[g.method == "single_model"]
        ref = s.loc[s.val_macro_f1.idxmax()]
        d = g[g.method == "device_only"]
        picks = [("Send every flow to the best detector", ref, 0.0), ("Device only", d.loc[d.val_macro_f1.idxmax()], 1.0)]
        for b in (0.1, 0.3):
            j = g[(g.method == "joint_nsga2") & (g.val_mean_cost <= b * ref.val_mean_cost)]
            if not j.empty:
                x = j.loc[j.val_macro_f1.idxmax()]
                picks.append((f"Joint, {100 * b:g}\\% budget", x, x.share_t1))
        rows += [dict(design=name, seed=seed, uj=x.mean_cost / 1e3, f1=x.macro_f1, minrec=x.minority_recall, fpr=x.fpr,
                      local=local) for name, x, local in picks]
    return pd.DataFrame(rows).groupby("design", sort=False).mean(numeric_only=True)


def scenario_table(cfg, out: Path) -> None:
    body = []
    for k, d in cfg.datasets.items():
        body.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{d.label}}}}} \\\\")
        for name, r in scenario_rows(d.cost, SCENARIO_PROFILE).iterrows():
            battery = 100 * r.uj * 1e-6 * FLOWS_PER_DAY * 365 / COIN_CELL_J
            body.append(f"{name} & {r.uj:,.1f} & {battery:.1f}\\% & {r.f1:.3f} & {r.minrec:.3f} & {100 * r.fpr:.1f}\\% & "
                        f"{r.fpr * FLOWS_PER_DAY:,.0f} & {100 * r.local:.0f}\\% \\\\")
    table(out, f"Deployment example with model-based energy projections: a BLE sensor on a coin cell (about {COIN_CELL_J / 1000:.1f}\\,kJ) assumed to produce "
               f"{FLOWS_PER_DAY:,} flows per day, device energy only (Table~\\ref{{tab:costs}}). Battery: share of the coin "
               "cell used per year by the IDS. False alarms: benign flows flagged per day if all flows were benign. "
               "Local: share of flows decided on the sensor. Deployments selected on validation; test metrics, mean "
               "over three seeds.", "tab:scenario", "@{}lrrrrrrr@{}",
          "Design & \\textmu J/flow & Battery/yr & Macro-F1 & Min.\\ rec. & FPR & False alarms/day & Local \\\\", body,
          wide=True)


OP_PROFILE = "system_ble"
OP_SCENARIOS = [("battery", "Battery-critical"), ("low_false_alarms", "Low false alarms"),
                ("rare_attacks", "Rare attacks"), ("balanced", "Balanced"),
                ("strict_false_alarms", "Strict false alarms"), ("secure_budget", "Secure on a budget")]
OP_METHODS = [("single_model", "Single detector"), ("gateway_server_c2t", "Fixed two-tier (C2T)"),
              ("tuned_two_tier", "Tuned two-tier"), ("greedy", "Greedy search"), ("joint_nsga2", "Joint optimisation")]


def operating_points_table(cfg, out: Path) -> None:
    """Reads operating_points_profile<OP_PROFILE>.csv written by experiments/operating_points.py into each cost run."""
    from eids.cascade.selection import SCENARIOS
    scenarios = {s.name: s for s in SCENARIOS}
    keys = list(cfg.datasets)
    ops = {k: run_csv(cfg.datasets[k].cost, f"operating_points_profile{OP_PROFILE}.csv") for k in keys}
    head = ("Scenario & Method & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys)
            + " \\\\\n" + " ".join(f"\\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(len(keys))) + "\n"
            + " & & " + " & ".join("Macro-F1 & Min.\\ rec. & FPR & DR & Energy" for _ in keys) + " \\\\")

    def violated(row, sc) -> bool:
        """Does a deployment chosen on validation break one of the scenario's constraints on the test split?"""
        ref = {"macro_f1": row.reference_macro_f1, "mean_cost": row.reference_cost}
        return (any(row[m] > v for m, v in sc.max_values.items())
                or any(row[m] < v for m, v in sc.min_values.items())
                or any(row[m] < f * ref[m] for m, f in sc.min_fraction_of_reference.items())
                or any(row[m] > f * ref[m] for m, f in sc.max_fraction_of_reference.items()))

    body = []
    for s, slabel in OP_SCENARIOS:
        for i, (m, mlabel) in enumerate(OP_METHODS):
            cells = []
            for k in keys:
                o = ops[k][(ops[k].scenario == s) & (ops[k].method == m)]
                if not o.feasible.all():
                    cells.append("\\multicolumn{5}{c}{infeasible}")
                    continue
                n_bad = sum(violated(r, scenarios[s]) for _, r in o.iterrows())
                mark = f"$^{{\\dagger{n_bad}}}$" if n_bad else ""
                energy = 100 * (o.mean_cost / o.reference_cost).mean()
                cells.append(f"{o.macro_f1.mean():.3f}{mark} & {o.minority_recall.mean():.3f} & {100 * o.fpr.mean():.1f}\\% & "
                             f"{100 * o.detection_rate.mean():.1f}\\% & {energy:.1f}\\%")
            body.append(f"{slabel if i == 0 else ''} & {mlabel} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    table(out, "Operating points under the BLE system-energy profile: for each operator scenario "
               "(Section~\\ref{sec:setup}), the deployment of each method selected on validation, with test metrics "
               "(mean over three seeds). DR: attack detection rate. Energy: share of the energy of sending every flow to "
               "the best single detector. Infeasible: no deployment of the method satisfies the scenario's constraints "
               "on the validation split for every seed. $\\dagger n$: the deployment selected on validation violates at "
               "least one of the scenario's constraints on the test split for $n$ of the three seeds.",
          "tab:operating", "@{}ll" + "rrrrr" * len(keys) + "@{}", head, body[:-1], wide=True)


TIERS = {"mcu": 0, "gateway": 1, "server": 2}
TIER_STYLE = [("Decided on the device", "#2a78d6"), ("Escalated to the gateway", "#eda100"),
              ("Escalated to the server", "#eb6834")]
CATEGORY_LABELS = {"BruteForce": "Brute force", "Recon": "Reconnaissance"}


def selected_deployment(run: str, profile: str, budget: float, seed: int = 0) -> pd.Series:
    """The joint deployment of the scenario table: best validation macro-F1 within the energy budget."""
    f = run_csv(run, "fronts.csv")
    g = f[(f.profile == profile) & (f.seed == seed)]
    s = g[g.method == "single_model"]
    ref = s.loc[s.val_macro_f1.idxmax()]
    j = g[(g.method == "joint_nsga2") & (g.val_mean_cost <= budget * ref.val_mean_cost)]
    return j.loc[j.val_macro_f1.idxmax()]


def routing(pool_dir: Path, dep: pd.Series) -> pd.DataFrame:
    """Share of test flows of each true category decided at each tier by a stored deployment."""
    from eids.cascade.simulate import Tier, run_cascade
    pool = load_pool(pool_dir)
    cats = np.array(load_json(pool_dir / "pool.json")["categories"])
    parts = [p.split(":") for p in dep.deployment.split(" -> ")]
    tiers = [Tier(t, pool.pred["test"][pool.names.index(m)].astype(np.int64),
                  pool.conf["test"][pool.names.index(m)].astype(np.float32), 0.0) for t, m in parts]
    thresholds = [np.array(t) for t in json.loads(dep.thresholds)]
    _, decided_at, _ = run_cascade(tiers, thresholds)
    tier_ids = np.array([TIERS[t] for t, _ in parts])[decided_at]
    true_cat = cats[pool.y["test"].astype(np.int64)]
    rows = [{"category": c, **{k: float(np.mean(tier_ids[true_cat == c] == k)) for k in range(3)}}
            for c in sorted(set(cats))]
    return pd.DataFrame(rows).set_index("category")


def routing_figure(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    fig, axes = plt.subplots(len(keys), 1, figsize=(3.45, 4.4), squeeze=False)
    routes = {k: routing(resolve_path(cfg.datasets[k].pool), selected_deployment(cfg.datasets[k].cost, SCENARIO_PROFILE, 0.3))
              for k in keys}
    used = [t for t in range(len(TIER_STYLE)) if any(r[t].sum() > 0 for r in routes.values())]  # no empty legend entries
    for ax, k in zip(axes[:, 0], keys):
        d = cfg.datasets[k]
        r = routes[k].sort_values(0)
        ys = np.arange(len(r))
        left = np.zeros(len(r))
        for t in used:
            label, color = TIER_STYLE[t]
            ax.barh(ys, r[t], left=left, color=color, height=0.7, label=label, edgecolor="white", linewidth=1)
            left += r[t].to_numpy()
        for y, v in zip(ys, r[0]):
            inside = v >= 0.15
            ax.text(v - 0.02 if inside else v + 0.02, y, f"{100 * v:.0f}%", va="center", ha="right" if inside else "left",
                    fontsize=7, color="white" if inside else INK)
        ax.set_yticks(ys, [CATEGORY_LABELS.get(c, c) for c in r.index], fontsize=7.5)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        ax.set_title(d.label, loc="left", color=INK, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(False)
    axes[-1, 0].set_xlabel("Share of test flows of the category", fontsize=8)
    handles, hl = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, hl, loc="upper center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def scenario_macros(cfg, out: Path) -> None:
    """Numbers of the deployment-example figure, as LaTeX macros, so the TikZ figure follows the results."""
    d = cfg.datasets[cfg.scenario_dataset]
    r = scenario_rows(d.cost, SCENARIO_PROFILE)
    battery = lambda x: 100 * x.uj * 1e-6 * FLOWS_PER_DAY * 365 / COIN_CELL_J  # noqa: E731
    rows = {"All": r.iloc[0], "Dev": r.iloc[1], "Joint": r.loc["Joint, 30\\% budget"]}
    lines = [f"\\newcommand{{\\scnDataset}}{{{d.label}}}", f"\\newcommand{{\\scnFlows}}{{{FLOWS_PER_DAY:,}}}"]
    for key, x in rows.items():
        lines += [f"\\newcommand{{\\scn{key}Battery}}{{{battery(x):.1f}\\%}}",
                  f"\\newcommand{{\\scn{key}FOne}}{{{x.f1:.2f}}}",
                  f"\\newcommand{{\\scn{key}Alarms}}{{{x.fpr * FLOWS_PER_DAY:,.0f}}}",
                  f"\\newcommand{{\\scn{key}Local}}{{{100 * x.local:.0f}}}"]
    dep = selected_deployment(d.cost, SCENARIO_PROFILE, 0.3)
    models = dict(p.split(":") for p in dep.deployment.split(" -> "))
    route = routing(resolve_path(d.pool), dep).drop(index="Benign")
    names = lambda idx: ", ".join(CATEGORY_LABELS.get(c, c).lower() if c not in ("DoS", "DDoS", "MITM") else c  # noqa: E731
                                  for c in idx)
    lines += [f"\\newcommand{{\\scnDeviceModel}}{{{describe(models['mcu'])}}}",
              f"\\newcommand{{\\scnUpperModel}}{{{describe(models.get('server') or models['gateway'])}}}",
              f"\\newcommand{{\\scnUpperTier}}{{{'Edge server' if 'server' in models else 'Gateway'}}}",
              f"\\newcommand{{\\scnKept}}{{{names(route[0].nlargest(3).index)}}}",
              f"\\newcommand{{\\scnEscalated}}{{{names(route[0].nsmallest(3).index)}}}"]
    write(out, "\n".join(lines) + "\n")


def describe(name: str) -> str:
    """Readable detector name for figures, e.g. decision_tree_l64_f20 -> decision tree, 64 leaves, 20 features."""
    import re
    if "tuned" in name:
        return "LightGBM, tuned"
    m = re.match(r"(decision_tree|random_forest|extra_trees|lightgbm)(?:_n(\d+))?_l(\d+|None)(?:_f(\d+))?", name)
    if not m:
        return name.replace("_", " ")
    kind, n, leaves, feats = m.groups()
    parts = [{"decision_tree": "decision tree", "random_forest": "random forest", "extra_trees": "Extra Trees",
              "lightgbm": "LightGBM"}[kind]]
    if n:
        parts.append(f"{n} trees")
    parts.append("unrestricted" if leaves == "None" else f"{leaves} leaves")
    if feats:
        parts.append(f"{feats} features")
    return ", ".join(parts)


EVASION_ALPHAS = [0.0, 0.2, 0.5]
EVASION_DEFENCES = [("searched", 0.0, "None (searched thresholds)"), ("0.9", 0.0, "Benign threshold 0.9"),
                    ("searched", 0.1, "Random audit of 10\\%")]


def evasion_table(cfg, out: Path) -> None:
    """Mimicry attack on the device tier of the battery deployment (experiments/evasion.py), with two defences."""
    keys = list(cfg.datasets)
    res = {k: run_csv(cfg.datasets[k].evasion, "results.csv") for k in keys}
    head = ("Defence & $\\alpha$ & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys)
            + " \\\\\n" + " ".join(f"\\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(keys))) + "\n"
            + " & & " + " & ".join("Escape & Missed & FPR & \\textmu J" for _ in keys) + " \\\\")
    body = []
    for th, audit, label in EVASION_DEFENCES:
        for i, a in enumerate(EVASION_ALPHAS):
            cells = []
            for k in keys:
                r = res[k]
                x = r[(r.benign_threshold.astype(str) == th) & np.isclose(r.audit, audit) & np.isclose(r.alpha, a)].iloc[0]
                cells.append(f"{100 * x.escape:.1f}\\% & {100 * x.missed:.1f}\\% & {100 * x.benign_fpr:.1f}\\% & "
                             f"{x.device_uj:.0f}")
            body.append(f"{label if i == 0 else ''} & {a:g} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    table(out, "Mimicry attack on the device tier of the battery deployment (device-battery BLE profile, 30\\% of the "
               "energy budget, seed 0): every attack flow is moved a fraction $\\alpha$ of the way towards the median "
               "benign flow on the 20 features the device reads. Escape: attacks accepted on the device as benign; "
               "Missed: attacks the whole cascade labels benign; FPR: benign flows raised as attacks; \\textmu J: device "
               "energy per flow. Test split.", "tab:evasion", "@{}lr" + "rrrr" * len(keys) + "@{}", head, body[:-1],
          wide=True)


FEATURE_UJ = 0.38  # extracting a ten-packet flow: ~100 cycles per packet at 0.3825 nJ per cycle (STM32L476)
OVERHEADS = [  # (label, feature-extraction factor, per-transmission factor, idle current in uA)
    ("Classification and transmission only", 0, 1, 0),
    ("+ feature extraction", 1, 1, 0),
    ("+ feature extraction $\\times 100$", 100, 1, 0),
    ("+ transmission overhead $\\times 3$", 1, 3, 0),
    ("+ idle current 2\\,\\textmu A", 1, 1, 2),
    ("+ idle current 10\\,\\textmu A", 1, 1, 10),
    ("Worst case: extraction $\\times 100$, idle 10\\,\\textmu A", 100, 1, 10),
]


def device_energy_parts(run: str) -> pd.DataFrame:
    """Per design (send every flow, device only, joint at 10% and 30% of the energy), mean over seeds of the
    device's computation energy per flow, its share of transmitted flows, and the test macro-F1 (device-battery
    BLE profile). Transmission energy per flow record is read from the cost run's configuration."""
    conf = load_json(resolve_path(run) / "config.json")
    radio_uj = conf["radio_nj"]["ble"] * conf.get("record_scale", 1.0) / 1000
    f = run_csv(run, "fronts.csv")
    rows = []
    for seed, g in f[f.profile == SCENARIO_PROFILE].groupby("seed"):
        s = g[g.method == "single_model"]
        ref = s.loc[s.val_macro_f1.idxmax()]
        d = g[g.method == "device_only"]
        picks = [("Send every flow", ref), ("Device only", d.loc[d.val_macro_f1.idxmax()])]
        for b in (0.1, 0.3):
            j = g[(g.method == "joint_nsga2") & (g.val_mean_cost <= b * ref.val_mean_cost)]
            picks.append((f"Joint, {100 * b:g}\\%", j.loc[j.val_macro_f1.idxmax()]))
        for name, x in picks:
            sent = 1.0 - x.share_t1 if x.deployment.startswith("mcu:") else 1.0
            rows.append(dict(design=name, seed=seed, sent=sent, compute_uj=x.mean_cost / 1000 - sent * radio_uj,
                             f1=x.macro_f1))
    out = pd.DataFrame(rows).groupby("design", sort=False).mean(numeric_only=True)
    out.attrs["radio_uj"] = radio_uj
    return out


def overhead_table(cfg, out: Path) -> None:
    """Battery life of each design when feature extraction, transmission overhead and idle current are added."""
    keys = list(cfg.datasets)
    parts = {k: device_energy_parts(cfg.datasets[k].cost) for k in keys}
    designs = list(parts[keys[0]].index)
    short = {"Send every flow": "Send all", "Device only": "Device", "Joint, 10\\%": "Joint 10\\%",
             "Joint, 30\\%": "Joint 30\\%"}
    head = ("Energy included & " + " & ".join(f"\\multicolumn{{{len(designs) + 1}}}{{c}}{{{cfg.datasets[k].label}}}"
                                           for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + (len(designs) + 1) * i}-{1 + (len(designs) + 1) * (i + 1)}}}"
                       for i in range(len(keys))) + "\n"
            + " & " + " & ".join(" & ".join(short[d] for d in designs) + " & Ratio" for _ in keys) + " \\\\")
    body = ["Macro-F1 & " + " & ".join(" & ".join(f"{parts[k].loc[d, 'f1']:.3f}" for d in designs) + " & "
                                       for k in keys) + " \\\\", "\\midrule"]
    for label, fx, tx, idle_ua in OVERHEADS:
        cells = []
        for k in keys:
            p = parts[k]
            per_flow = fx * FEATURE_UJ + p.compute_uj + p.sent * p.attrs["radio_uj"] * tx  # uJ
            mj_per_day = FLOWS_PER_DAY * per_flow * 1e-3 + idle_ua * 1e-6 * 3.0 * 86400 * 1e3
            cells.append(" & ".join(f"{mj_per_day[d]:,.0f}" for d in designs)
                         + f" & {mj_per_day['Send every flow'] / mj_per_day['Joint, 30\\%']:.1f}$\\times$")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table(out, f"Projected daily energy (mJ, model-based estimate) of a BLE sensor producing {FLOWS_PER_DAY:,} flows per day when energy beyond "
               "classification and transmission is included: feature extraction (estimated at "
               f"{FEATURE_UJ}\\,\\textmu J per flow, Section~\\ref{{sec:discussion}}), per-transmission overhead "
               "(acknowledgements, retransmissions, wake-up), and a constant idle current of the sensor and its radio "
               "link at 3\\,V. Joint deployments are those of Table~\\ref{tab:scenario}; Ratio: daily energy of "
               "sending every flow over that of the joint deployment at 30\\% of the energy, equal to the ratio of "
               "their battery lives. Mean over three seeds.",
          "tab:overhead", "@{}l" + ("r" * (len(designs) + 1)) * len(keys) + "@{}", head, body, wide=True)


PREVALENCES = [0.5, 0.1, 0.01]


def operating_deployment(cost_run: str, scenario: str, method: str, seed: int = 0) -> pd.Series:
    """Front row (with thresholds) of the deployment that operating_points.py selected for a scenario."""
    op = run_csv(cost_run, f"operating_points_profile{OP_PROFILE}.csv")
    pick = op[(op.scenario == scenario) & (op.method == method) & (op.seed == seed)].iloc[0]
    f = run_csv(cost_run, "fronts.csv")
    rows = f[(f.profile == OP_PROFILE) & (f.seed == seed) & (f.method == method) & (f.deployment == pick.deployment)]
    return rows.iloc[0]


def cascade_predictions(pool_dir: Path, dep: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(true class, predicted class, category of each class) on the test split for a stored deployment."""
    from eids.cascade.simulate import Tier, run_cascade
    pool = load_pool(pool_dir)
    parts = [p.split(":") for p in dep.deployment.split(" -> ")]
    tiers = [Tier(t, pool.pred["test"][pool.names.index(m)].astype(np.int64),
                  pool.conf["test"][pool.names.index(m)].astype(np.float32), 0.0) for t, m in parts]
    thresholds = [np.array(t) for t in json.loads(dep.thresholds)] if len(parts) > 1 else []
    pred, _, _ = run_cascade(tiers, thresholds) if thresholds else (tiers[0].pred, None, None)
    cats = np.array(load_json(pool_dir / "pool.json")["categories"])
    return pool.y["test"].astype(np.int64), pred, cats


def category_table(cfg, out: Path) -> None:
    """Per attack category: share detected as an attack and share assigned the correct class, for the best single
    detector and for the battery-critical joint deployment (system energy, BLE; seed 0)."""
    keys = list(cfg.datasets)
    designs = [("single_model", "battery", "Best single"), ("joint_nsga2", "battery", "Joint, battery")]
    results = {}
    for k in keys:
        d = cfg.datasets[k]
        for method, scenario, _ in designs:
            dep = operating_deployment(d.cost, scenario, method)
            if method == "single_model":  # the reference detector itself, not the battery pick
                f = run_csv(d.cost, "fronts.csv")
                g = f[(f.profile == OP_PROFILE) & (f.seed == 0) & (f.method == "single_model")]
                dep = g.loc[g.val_macro_f1.idxmax()]
            y, pred, cats = cascade_predictions(resolve_path(d.pool), dep)
            benign_idx = np.flatnonzero(cats == "Benign")
            results[(k, method)] = {c: (float(np.mean(~np.isin(pred[cats[y] == c], benign_idx))),
                                        float(np.mean(pred[cats[y] == c] == y[cats[y] == c])))
                                    for c in sorted(set(cats))}
    cats_all = sorted({c for per in results.values() for c in per}, key=lambda c: (c != "Benign", c))
    head = ("Category & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(keys))) + "\n"
            + " & " + " & ".join(" & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for _, _, lab in designs) for _ in keys)
            + " \\\\\n & " + " & ".join("Det. & Class & Det. & Class" for _ in keys) + " \\\\")
    body = []
    for c in cats_all:
        cells = []
        for k in keys:
            for method, _, _ in designs:
                v = results[(k, method)].get(c)
                cells.append("-- & --" if v is None else f"{100 * v[0]:.1f} & {100 * v[1]:.1f}")
        body.append(f"{CATEGORY_LABELS.get(c, c)} & " + " & ".join(cells) + " \\\\")
    table(out, "Detection per category (test split, seed 0) for the best single detector and the battery-critical joint "
               "deployment of Table~\\ref{tab:operating}. Det.: \\% of the category's flows labelled as any attack (for "
               "benign traffic: the false-positive rate); Class: \\% labelled with their exact class.",
          "tab:category", "@{}l" + "rrrr" * len(keys) + "@{}", head, body, wide=True)


def prevalence_table(cfg, out: Path) -> None:
    """Share of alerts that are real attacks when attacks are rarer than in the test data (Bayes' rule on the
    measured detection and false-positive rates)."""
    keys = list(cfg.datasets)
    designs = [("device_only", None, "Device only"), ("single_model", None, "Best single detector"),
               ("joint_nsga2", "battery", "Joint, battery-critical"), ("joint_nsga2", "low_false_alarms",
                                                                        "Joint, low false alarms")]
    head = ("Design & " + " & ".join(f"\\multicolumn{{{len(PREVALENCES)}}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys)
            + " \\\\\n" + " ".join(f"\\cmidrule(lr){{{2 + len(PREVALENCES) * i}-{1 + len(PREVALENCES) * (i + 1)}}}"
                                   for i in range(len(keys)))
            + "\n & " + " & ".join(" & ".join(f"{100 * p:g}\\%" for p in PREVALENCES) for _ in keys) + " \\\\")
    body = []
    for method, scenario, label in designs:
        cells = []
        for k in keys:
            d = cfg.datasets[k]
            f = run_csv(d.cost, "fronts.csv")
            precision = []
            for seed in sorted(f.seed.unique()):
                if scenario:
                    row = operating_deployment(d.cost, scenario, method, seed)
                else:
                    g = f[(f.profile == OP_PROFILE) & (f.seed == seed) & (f.method == method)]
                    row = g.loc[g.val_macro_f1.idxmax()]
                dr, fpr = row.detection_rate, row.fpr
                precision.append([p * dr / (p * dr + (1 - p) * fpr) for p in PREVALENCES])
            cells += [f"{100 * v:.0f}\\%" for v in np.mean(precision, axis=0)]
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    table(out, "Precision of alerts (share of alerts that are real attacks) if attacks made up 50\\%, 10\\% or 1\\% "
               "of the flows, computed from each design's test detection rate and benign false-positive rate (mean over three seeds, "
               "BLE system-energy profile). Real networks usually see far fewer attacks than the benchmark datasets.",
          "tab:prevalence", "@{}l" + "r" * len(PREVALENCES) * len(keys) + "@{}", head, body, wide=True)


TIMING_ROWS = [
    ("decision_tree_l256_f20", "Decision tree, 256 leaves, 20 features"),
    ("decision_tree_l1024", "Decision tree, 1,024 leaves"),
    ("random_forest_n10_l512", r"Random forest, $10\times512$"),
    ("extra_trees_n10_l512", r"Extra Trees, $10\times512$"),
    ("random_forest_n100_lNone_unw", "Random forest, 100 unrestricted trees"),
    ("lightgbm_n200_l32", r"LightGBM, $200\times32$"),
    (r"lightgbm_n\d+_l\d+_tuned_f1", "LightGBM, tuned"),
    ("mlp_32_f20", "MLP, 32 hidden units, 20 features"),
]


def timing_table(cfg, out: Path) -> None:
    keys = list(cfg.datasets)
    t = {k: run_csv(cfg.timing, f"timing_{k}.csv") for k in keys}
    head = ("Detector & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{cfg.datasets[k].label}}}" for k in keys) + " \\\\\n"
            + " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(keys))) + "\n"
            + " & " + " & ".join("Train (s) & \\textmu s/flow & Ops & Size (KB)" for _ in keys) + " \\\\")
    body = []
    for name, label in TIMING_ROWS:
        cells = []
        for k in keys:
            r = t[k][t[k].model.str.fullmatch(name)].iloc[0]
            cells.append(f"{r.fit_s:,.1f} & {r.us_per_flow:.2f} & {fmt_int(r.ops)} & {r.size_kb:,.1f}")
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    rows = {k: fmt_int(t[k].train_rows.iloc[0]) for k in keys}
    table(out, "Measured training time and inference time of representative detectors (seed 0; "
               + ", ".join(f"{rows[k]} training records for {cfg.datasets[k].label}" for k in keys) + "). Training "
               "uses all 20 threads of an Intel Core i5-13500HX laptop CPU (14 cores); inference runs on one thread over 20,000 test flows and is reported "
               "per flow. Python inference times include library overhead and are not representative of device "
               "firmware; the operation counts are the cost model used in the search.",
          "tab:timing", "@{}l" + "rrrr" * len(keys) + "@{}", head, body, wide=True)


# ---------------------------------------------------------------- audits and diagnostics

def edge_table(run: str, out: Path) -> None:
    r = run_csv(run, "results.csv")
    st = load_json(resolve_path(run) / "stats.json")
    labels = {"counters_only": "Counters only$^{a}$",
              "behavioural_only": "Behavioural features only",
              "both_standard_protocol": "Both (usual protocol)"}
    g = r.groupby("features")[["accuracy", "macro_f1", "n_columns"]].agg(["mean", "std"])
    body = [f"{labels[f]} & {int(g.loc[f, ('n_columns', 'mean')])} & {pm(g.loc[f, ('accuracy', 'mean')], g.loc[f, ('accuracy', 'std')])} & "
            f"{pm(g.loc[f, ('macro_f1', 'mean')], g.loc[f, ('macro_f1', 'std')])} \\\\" for f in labels]
    table(out, f"Edge-IIoTset leakage audit: LightGBM after the standard drop list and deduplication "
               f"({fmt_int(st['rows_after_standard_drop_and_dedup'])} records; test split, three seeds). Without the "
               f"counter columns only {fmt_int(st['distinct_rows_without_counters'])} records remain distinct "
               f"({100 * st['share_duplicates_without_counters']:.1f}\\% duplicates). "
               "$^{a}$TCP sequence and acknowledgement numbers, checksums and stream indices.",
          "tab:edge", "@{}lrcc@{}", "Features & Columns & Accuracy & Macro-F1 \\\\", body)


def replication_table(cfg, out: Path) -> None:
    body = []
    for k, d in cfg.datasets.items():
        r = run_csv(d.replication, "results.csv")
        st = load_json(resolve_path(d.replication) / "stats.json")
        g = r.groupby(["protocol", "model"]).mean(numeric_only=True)
        for i, ((protocol, model), row) in enumerate(g.iterrows()):
            name = f"\\multirow{{{len(g)}}}{{*}}{{{d.label}}}" if i == 0 else ""
            pname = "published" if protocol.startswith("published") else "ours"
            mname = "RF" if "forest" in model else "LightGBM"
            body.append(f"{name} & {pname} & {mname} & {row.fine_accuracy:.3f} & {row.fine_macro_f1:.3f} & "
                        f"{row.category_macro_f1:.3f} & {row.binary_accuracy:.3f} \\\\")
        body.append("\\midrule")
    table(out, "Published versus our evaluation protocol with the same default, unweighted models (mean over three seeds). "
               "Published: raw records with duplicates kept, 500,000-record sample, random 80/20 split. Ours: the cleaned "
               "data of Table~\\ref{tab:datasets}.", "tab:replication", "@{}lllcccc@{}",
          "Dataset & Protocol & Model & Fine acc. & Fine macro-F1 & Category macro-F1 & Binary acc. \\\\", body[:-1])


def escalation_table(run: str, out: Path) -> None:
    sc = run_csv(run, "scores.csv")
    mi = run_csv(run, "missed_attacks.csv")
    names = {"decision_tree_l256_f20": "Tree, 256 leaves, 20 features", "decision_tree_l1024_f20": "Tree, 1,024 leaves, 20 features",
             "random_forest_n10_l128": r"Forest, $10\times128$"}
    body = []
    for _, r in sc.iterrows():
        m = mi[mi.detector == r.detector].set_index("tau")
        body.append(f"{names.get(r.detector, r.detector)} & {r.auroc_max_probability:.3f} & {r.auroc_margin:.3f} & {r.auroc_neg_entropy:.3f} & "
                    f"{r.ece_raw:.3f} / {r.ece_isotonic:.3f} & {100 * m.loc[0.0, 'attacks_kept_as_benign']:.1f}\\% / "
                    f"{100 * m.loc[0.5, 'attacks_kept_as_benign']:.1f}\\% / {100 * m.loc[0.7, 'attacks_kept_as_benign']:.1f}\\% \\\\")
    table(out, "Escalation diagnostics for device-tier detectors (CIC-IoT2023, test split). AUROC: separation of wrong from "
               "correct predictions by each uncertainty score. ECE: expected calibration error before / after isotonic "
               "calibration. Missed: share of attacks predicted benign with confidence at least $\\tau$ and kept on the device.",
          "tab:escalation", "@{}lccccc@{}",
          "Detector & AUROC conf. & AUROC margin & AUROC entropy & ECE & Missed at $\\tau=0/0.5/0.7$ \\\\", body, wide=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_artifacts.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    style()
    paper = resolve_path(cfg.paper_dir)
    T, F = paper / "tables", paper / "figures"

    datasets_table(cfg, T / "datasets.tex")
    ref = cfg.datasets[cfg.confidence_figure_dataset]
    class_distribution_figure(load_json(resolve_path(ref.processed) / "meta.json"), F / "class_distribution.pdf")
    pools = {k: pool_metrics(load_pool(resolve_path(d.pool))) for k, d in cfg.datasets.items()}
    pool_table(cfg, pools, T / "pool.tex")
    quality_cost_figure(cfg, pools, F / "quality_vs_cost.pdf")
    confidence_by_category_figure(load_pool(resolve_path(ref.pool)), resolve_path(ref.pool), F / "confidence_by_category.pdf")

    sweeps = main_results(cfg)
    main_table(cfg, sweeps, T / "main_results.tex")
    hv_tx_figure(cfg, sweeps, F / "hv_vs_tx.pdf")
    fronts_figure(cfg, F / "pareto_fronts.pdf")
    design_table(cfg, T / "design.tex")
    ladder_table(cfg, T / "ladder.tex")
    cost_table(cfg, T / "cost_sensitivity.tex")
    cost_figure(cfg, F / "cost_sensitivity.pdf")
    budget_table(cfg, T / "budget.tex")
    budget_figure(cfg, F / "energy_budget.pdf")
    scenario_table(cfg, T / "scenario.tex")
    operating_points_table(cfg, T / "operating_points.tex")
    category_table(cfg, T / "category.tex")
    prevalence_table(cfg, T / "prevalence.tex")
    overhead_table(cfg, T / "overhead.tex")
    evasion_table(cfg, T / "evasion.tex")
    timing_table(cfg, T / "timing.tex")
    routing_figure(cfg, F / "routing_by_category.pdf")
    scenario_macros(cfg, F / "scenario_numbers.tex")

    study = run_csv(cfg.search_study, "results.csv")
    search_study_table(study, T / "search_study.tex")
    search_curve_figure(study, F / "search_curves.pdf")
    edge_table(cfg.edge_audit, T / "edge_audit.tex")
    replication_table(cfg, T / "replication.tex")
    escalation_table(cfg.escalation_diagnostics, T / "escalation.tex")


if __name__ == "__main__":
    main()
