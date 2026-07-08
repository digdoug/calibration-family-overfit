"""
Bridge: ControlArena per-family eval logs -> TransferScores.

The W2 pipeline:

  (per family F)
    inspect_ai eval -> .eval logs
       --[.venv-controlarena]-->
    w1_gate1_extract.py -> per-family samples.csv
       --[.venv (analysis lock surface)]-->
    THIS MODULE: per_family_csv_to_long + build_transfer_scores_from_per_family
       -> TransferScores -> build_transfer_matrices

Why the CSV intermediate: keeps the headline statistic on the locked
analysis venv (numpy/sklearn/statsmodels pinned in requirements.lock.txt),
while the agent runtime (control-arena + Inspect AI + dozens of agent
deps) lives in the heavier .venv-controlarena. The CSV is the audited
hand-off (the design doc s.10 -- generate-once / score-many).

Task-split assignment is GLOBAL (the design doc s.4): same task_id maps to the
same split across families, deterministic in (seed, task_id). This is what
makes "calibrate-on-family-A's calibration-benign, test-on-family-B's
test-sabotage" a clean experiment rather than an entanglement of split-
choice and family-choice.

Monitor-kind matters (the design doc s.1a + s.10):
  - fixed_prompt: a single scoring function applied to every (family, task,
    label). The TransferScores schema replicates the same score across all
    calib_family values for each (task, eval_family, label, split) cell --
    runtime-asserted invariant. AUROC has no calibration axis.
  - family_conditioned: a per-calibration-family scoring function (few-shot
    seeded on family A, frozen-embedding head trained on family A, or QLoRA
    family A). Real per-cell scores; AUROC matrix is well-defined. Not
    implemented in this bridge yet -- that's the W7-8 stretch path.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transfer_data import (  # noqa: E402
    MonitorKind,
    TransferScores,
)


SplitName = Literal["train", "calibration", "test"]


def assign_splits_deterministic(
    task_ids: list[str],
    calibration_fraction: float = 0.6,
    test_fraction: float = 0.4,
    train_fraction: float = 0.0,
    seed: int = 0,
) -> dict[str, SplitName]:
    """Deterministic GLOBAL task -> split mapping (the design doc s.4).

    Same task_id -> same split across families (paired transcripts). Uses
    SHA-256 of (seed, task_id) for stability across machines / Python
    versions; do not switch hash without bumping the provenance seed.
    """
    total = calibration_fraction + test_fraction + train_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"split fractions must sum to 1.0; got {total} "
            f"(calibration={calibration_fraction}, test={test_fraction}, "
            f"train={train_fraction})"
        )

    out: dict[str, SplitName] = {}
    for tid in sorted(set(task_ids)):
        h = hashlib.sha256(f"{seed}::{tid}".encode()).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF
        if u < calibration_fraction:
            out[tid] = "calibration"
        elif u < calibration_fraction + test_fraction:
            out[tid] = "test"
        else:
            out[tid] = "train"
    return out


def per_family_csv_to_long(
    csv_path: Path,
    family_name: str,
    task_split: dict[str, SplitName],
    success_filter: bool = True,
) -> pd.DataFrame:
    """Load per-family extracted CSV (from w1_gate1_extract.py-style
    extraction in .venv-controlarena) and convert to the long-form
    {task_id, family, split, label, score} rows the bridge expects.

    Input CSV columns (from samples_df-based extract):
        sample_id, task_id, mode, main_task_success, side_task_success,
        suspicion, log_path

    Mapping:
        mode='honest' -> label='benign'
        mode='attack' -> label='sabotage'

    Success filter (the design doc s.4): "benign = main task succeeds; sabotage =
    main AND side task succeed (don't credit monitor for catching failed
    attacks)." Enabled by default. Set success_filter=False for diagnostic
    runs where you want to see ALL trajectories (e.g., the W1 contrast
    where the basic_monitor produced 0.0 across the board).
    """
    df = pd.read_csv(csv_path)
    df["task_id"] = df["task_id"].astype(str)
    df["family"] = family_name
    df["label"] = df["mode"].map({"honest": "benign", "attack": "sabotage"})
    df["split"] = df["task_id"].map(task_split)
    # parse booleans defensively (samples_df gives them as nullable booleans,
    # CSV round-trip can land as bool / object / NA)
    for col in ("main_task_success", "side_task_success"):
        if col in df.columns:
            df[col] = df[col].map(_to_bool)

    df = df.dropna(subset=["split", "label", "suspicion"])

    if success_filter:
        ben_mask = (df["label"] == "benign") & (df["main_task_success"] == True)  # noqa: E712
        sab_mask = (
            (df["label"] == "sabotage")
            & (df["main_task_success"] == True)
            & (df["side_task_success"] == True)
        )
        df = df[ben_mask | sab_mask].copy()

    df = df.rename(columns={"suspicion": "score"})
    return df[["task_id", "family", "split", "label", "score"]].reset_index(drop=True)


def _to_bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NA
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        if s in ("", "nan", "<na>", "none", "null"):
            return pd.NA
    if isinstance(v, (int, float)):
        return bool(v)
    return pd.NA


def build_transfer_scores_from_per_family(
    per_family: dict[str, pd.DataFrame],
    monitor_id: str,
    monitor_kind: MonitorKind,
    families: tuple[str, ...],
    generation_seed: int = 0,
) -> TransferScores:
    """Assemble per-family long-form scores into a TransferScores.

    `per_family` maps eval_family -> DataFrame with columns
    (task_id, family, split, label, score). Family in df must match the key.

    For monitor_kind='fixed_prompt':
        The monitor's scoring function is calib-invariant by definition
        (the design doc s.1a). The bridge replicates each (task, eval_family,
        split, label) score across every calib_family value, producing a
        column-constant grid. The TransferScores constructor asserts that
        column-constancy holds.

    For monitor_kind='family_conditioned':
        Not yet supported here. A future helper (W7-8) will take a 2-level
        dict {calib_family -> {eval_family -> df}} where the calib-A-tuned
        monitor's scores on family-B trajectories appear under
        (calib=A, eval=B), and the constructor's variance check confirms
        the monitor actually depends on calib_family.
    """
    if monitor_kind == "family_conditioned":
        raise NotImplementedError(
            "family_conditioned bridge is the W7-8 path (QLoRA / few-shot / "
            "frozen-embedding head). Use build_transfer_scores_from_per_cell."
        )

    if set(per_family.keys()) - set(families):
        raise ValueError(
            f"per_family contains eval_family values not in declared families: "
            f"{set(per_family.keys()) - set(families)}"
        )
    if set(families) - set(per_family.keys()):
        missing = set(families) - set(per_family.keys())
        raise ValueError(f"per_family is missing eval_family values: {missing}")

    rows = []
    for ev_family in families:
        df = per_family[ev_family]
        if df.empty:
            raise ValueError(f"No rows for eval_family={ev_family} (empty df)")
        bad_fam = df[df["family"] != ev_family]
        if not bad_fam.empty:
            raise ValueError(
                f"per_family[{ev_family}] contains rows with family != {ev_family}: "
                f"{sorted(set(bad_fam['family']))}"
            )
        for _, r in df.iterrows():
            for calib_family in families:
                rows.append({
                    "task_id": str(r["task_id"]),
                    "calib_family": calib_family,
                    "eval_family": ev_family,
                    "split": r["split"],
                    "label": r["label"],
                    "score": float(r["score"]),
                    "monitor_id": monitor_id,
                    "generation_seed": generation_seed,
                })

    return TransferScores(
        df=pd.DataFrame(rows),
        calib_families=families,
        eval_families=families,
        monitor_id=monitor_id,
        monitor_kind=monitor_kind,
    )
