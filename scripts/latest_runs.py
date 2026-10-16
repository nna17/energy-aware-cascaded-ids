"""Point every run folder in the artifact config at the newest run of the same experiment.

Each run folder is results/runs/<experiment>/<timestamp>; after re-running the experiments, this replaces every
timestamp with the newest one under results/runs/<experiment>, keeping the rest of the file (and its comments).

    python scripts/latest_runs.py --config configs/paper_artifacts.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from eids.utils.config import resolve_path

RUN = re.compile(r"(results/runs/([\w.-]+))/(\d{8}-\d{6})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_artifacts.yaml")
    args = parser.parse_args()
    path = Path(args.config)
    text = path.read_text(encoding="utf-8")

    def newest(m: re.Match) -> str:
        runs = sorted(p.name for p in resolve_path(m.group(1)).glob("*") if re.fullmatch(r"\d{8}-\d{6}", p.name))
        if not runs:
            print(f"no runs of {m.group(2)}; keeping {m.group(3)}")
            return m.group(0)
        if runs[-1] != m.group(3):
            print(f"{m.group(2)}: {m.group(3)} -> {runs[-1]}")
        return f"{m.group(1)}/{runs[-1]}"

    path.write_text(RUN.sub(newest, text), encoding="utf-8")


if __name__ == "__main__":
    main()
