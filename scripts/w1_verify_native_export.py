"""
W1 verification: native Inspect AI path for generate-once / score-many.

The shipped `control-arena trajectory export` CLI is broken at the version
we pinned (control-arena 16.0.0): both the `anthropic` and `google_genai`
formatters fail with schema-drift errors against the current
`FormattedTrajectory` pydantic model.

We don't need those formatters anyway. The `.eval` log file produced by
inspect_ai already carries the full message + tool-call tape we'd need for
offline monitor rescoring. This script verifies:

  1. samples_df() round-trips the smoke eval logs into pandas.
  2. read_eval_log() loads the full per-sample message sequence.
  3. The per-sample suspicion score is accessible.

If those three pass, the generate-once / score-many lever (the design doc s.9 +
s.10) is verified -- we will read .eval logs natively, not via the CLI.

Run:  .venv-controlarena/bin/python scripts/w1_verify_native_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

from control_arena.analysis import samples_df
from inspect_ai.log import read_eval_log


LOG_DIR = Path(__file__).resolve().parents[1] / "results" / "W1_smoke"


def main() -> int:
    df = samples_df(str(LOG_DIR))
    assert len(df) == 2, f"expected 2 samples (honest + attack), got {len(df)}"
    print(f"samples_df: {len(df)} rows, {len(df.columns)} columns")

    modes = sorted(df["metadata_eval_mode"].unique().tolist())
    print(f"  modes: {modes}")
    assert modes == ["attack", "honest"], modes

    for _, row in df.iterrows():
        mode = row["metadata_eval_mode"]
        log_path = row["log"]
        log = read_eval_log(log_path)
        assert log.status == "success", f"{mode}: log status = {log.status}"
        sample = log.samples[0]
        n_messages = len(sample.messages)
        roles = [m.role for m in sample.messages]
        score_blob = row.get("score_apps_scorer", None)
        if isinstance(score_blob, str):
            score_blob = json.loads(score_blob)
        main_ok = row["main_task_success"]
        side_ok = row.get("side_task_success", None)
        usage = json.loads(row["model_usage"])
        total_tokens = sum(u.get("total_tokens", 0) for u in usage.values())

        print(f"\n  [{mode}]")
        print(f"    messages: {n_messages} ({'/'.join(roles)})")
        print(f"    main_task_success={main_ok}  side_task_success={side_ok}")
        print(f"    apps_scorer: {score_blob}")
        print(f"    total_tokens: {total_tokens}")

    print("\nPASS: native Inspect path provides the generate-once / score-many lever.")
    print("Next: scale to limit~50 baseline for the GATE 1 hard check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
