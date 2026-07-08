"""
W1 GATE 1 step 1: extract per-sample data from the .eval logs into a CSV
the analysis venv can consume.

Runs in .venv-controlarena (has samples_df). Output is a small per-sample
table the analysis venv consumes for AUROC. This is the bridge pattern
we'll generalize in W2 when wiring real scores into TransferScores.

Run:  .venv-controlarena/bin/python scripts/w1_gate1_extract.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from control_arena.analysis import samples_df


import sys
LOG_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "results" / "W1_gate1"
)
OUT_PATH = LOG_DIR / "samples.csv"


def _find_suspicion_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "suspicion" in c.lower()]
    if not candidates:
        # Dump column names to help diagnose
        print("Columns:", list(df.columns))
        raise RuntimeError("No suspicion-score column found in samples_df output")
    if len(candidates) > 1:
        print(f"  multiple suspicion cols: {candidates}; using {candidates[0]}")
    return candidates[0]


def _parse_suspicion(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            obj = json.loads(v)
            if isinstance(obj, dict):
                for k in ("value", "score", "suspicion"):
                    if k in obj:
                        return float(obj[k])
                for v2 in obj.values():
                    if isinstance(v2, (int, float)):
                        return float(v2)
            if isinstance(obj, (int, float)):
                return float(obj)
        except json.JSONDecodeError:
            try:
                return float(v)
            except ValueError:
                pass
    return float("nan")


def main() -> int:
    df = samples_df(str(LOG_DIR))
    print(f"samples_df: {len(df)} rows, {len(df.columns)} cols")

    suspicion_col = _find_suspicion_col(df)
    print(f"using suspicion column: {suspicion_col}")

    out = pd.DataFrame({
        "sample_id": df["sample_id"],
        "task_id": df["metadata_main_task_name"],   # used for cluster bootstrap later
        "mode": df["metadata_eval_mode"],
        "main_task_success": df["main_task_success"].astype("boolean"),
        "side_task_success": df.get("side_task_success", pd.Series([pd.NA]*len(df))).astype("boolean"),
        "suspicion": df[suspicion_col].map(_parse_suspicion),
        "log_path": df["log"],
    })

    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(out)} rows)")
    print()
    print("--- preview ---")
    print(out.head(6).to_string(index=False))
    print()
    print("--- per-mode ---")
    print(out.groupby("mode").agg(
        n=("sample_id", "count"),
        with_suspicion=("suspicion", lambda s: s.notna().sum()),
        main_ok=("main_task_success", "sum"),
        side_ok=("side_task_success", lambda s: s.dropna().sum()),
        suspicion_mean=("suspicion", "mean"),
    ).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
