"""Introduction figure: detection quality against energy per flow for the main designs (BLE, system energy).

Reuses the budget curves of make_paper_artifacts.py, so it shows the same numbers as the energy-budget table.
    PYTHONPATH=src python scripts/make_teaser_figure.py --out ../paper_jnsm/figures/teaser.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_paper_artifacts as mpa  # noqa: E402

JOINT = ("Jointly optimised cascade", "#1baf7a", "D")
FIXED = ("Fixed two-tier design", "#eb6834", "s")
DEVICE = "#2a78d6"  # same hue as single models elsewhere in the paper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_artifacts.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = mpa.load_config(args.config)
    mpa.style()
    keys = list(cfg.datasets)
    fig, axes = plt.subplots(1, len(keys), figsize=(6.6, 2.8), squeeze=False)
    for j, k in enumerate(keys):
        ax = axes[0, j]
        r = mpa.budget_curves(cfg.datasets[k], "system_ble")
        nseeds = r.seed.nunique()
        ref, floor = r.ref_f1.mean(), r.floor_f1.mean()
        span = ref - floor
        # tight-budget region: budgets below the first one at which the fixed two-tier design fits on every seed
        fx = r[r.method == "gateway_server_c2t"].groupby("budget").size()
        tight = 100 * fx[fx == nseeds].index.min()
        ax.axvspan(0.1, tight, color="#ececea", zorder=0, linewidth=0)
        ax.text(0.13, ref - 0.1 * span, "tight budgets", fontsize=7, color=mpa.INK_2, va="top")
        ax.axhline(ref, color=mpa.INK_2, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.text(70, ref + 0.04 * span, "every flow sent to the best model", fontsize=7, color=mpa.INK_2,
                va="bottom", ha="right")
        ax.axhline(floor, color=DEVICE, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
        ax.text(90, floor + 0.03 * span, "device only", fontsize=7, color=mpa.INK_2, va="bottom", ha="right")
        for m, (label, color, marker) in (("gateway_server_c2t", FIXED), ("joint_nsga2", JOINT)):
            g = r[r.method == m].groupby("budget").f1
            n = g.size()
            g = g.mean()[n == nseeds]
            ax.plot(100 * g.index, g.values, color=color, marker=marker, markersize=5, linewidth=2,
                    markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
            if m == "joint_nsga2":
                ax.annotate("joint", (1, g.loc[0.01]), xytext=(6, -4), textcoords="offset points", fontsize=7,
                            color=mpa.INK)
            else:
                ax.annotate("fixed two-tier", (100 * g.index[0], g.values[0]), xytext=(2, -11),
                            textcoords="offset points", fontsize=7, color=mpa.INK)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 100)
        ax.set_ylim(floor - 0.08 * span, ref + 0.2 * span)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}%"))
        ax.set_title(cfg.datasets[k].label, loc="left", color=mpa.INK)
        ax.set_xlabel("Energy per flow (% of sending every flow)")
        if j == 0:
            ax.set_ylabel("Detection quality (macro-F1)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
