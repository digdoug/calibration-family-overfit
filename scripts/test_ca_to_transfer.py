"""
Tests for the ControlArena -> TransferScores bridge.

Uses the existing W1 GATE 1 v2 CSV (results/W1_gate1_v2/samples.csv) as a
real input fixture, since it has the canonical samples_df schema. Also
constructs purely-synthetic inputs for the schema guards.

Pytest entrypoint:
    .venv/bin/python -m pytest scripts/test_ca_to_transfer.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ca_to_transfer import (
    assign_splits_deterministic,
    build_transfer_scores_from_per_family,
    per_family_csv_to_long,
)
from transfer_data import (
    TransferScores,
    build_transfer_matrices,
    decompose_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
W1_CSV = REPO_ROOT / "results" / "W1_gate1_v2" / "samples.csv"


# ---------------------------------------------------------------------------
# assign_splits_deterministic
# ---------------------------------------------------------------------------
def test_split_assignment_is_deterministic_in_seed():
    ids = [str(i) for i in range(200)]
    a = assign_splits_deterministic(ids, seed=0)
    b = assign_splits_deterministic(ids, seed=0)
    assert a == b


def test_split_assignment_changes_with_seed():
    ids = [str(i) for i in range(200)]
    a = assign_splits_deterministic(ids, seed=0)
    b = assign_splits_deterministic(ids, seed=1)
    # Most ids should land in different splits across different seeds.
    differ = sum(1 for k in ids if a[k] != b[k])
    assert differ > 50


def test_split_proportions_match_fractions():
    ids = [str(i) for i in range(5000)]
    out = assign_splits_deterministic(ids, calibration_fraction=0.6, test_fraction=0.4)
    n_cal = sum(1 for v in out.values() if v == "calibration")
    n_test = sum(1 for v in out.values() if v == "test")
    assert abs(n_cal / 5000 - 0.6) < 0.02
    assert abs(n_test / 5000 - 0.4) < 0.02


def test_split_assignment_rejects_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits_deterministic(["a"], calibration_fraction=0.5, test_fraction=0.4)


# ---------------------------------------------------------------------------
# per_family_csv_to_long  (against W1 GATE 1 v2 data if present)
# ---------------------------------------------------------------------------
def _fixture_csv() -> Path:
    if not W1_CSV.exists():
        pytest.skip(f"{W1_CSV} not present; run scripts/w1_gate1_extract.py results/W1_gate1_v2")
    return W1_CSV


def test_per_family_csv_to_long_preserves_w1_scores_unfiltered():
    csv = _fixture_csv()
    raw = pd.read_csv(csv)
    task_ids = raw["task_id"].astype(str).unique().tolist()
    splits = {t: "test" for t in task_ids}   # put everything in test for the smoke
    out = per_family_csv_to_long(csv, "qwen", splits, success_filter=False)
    assert set(out["label"].unique()) <= {"benign", "sabotage"}
    assert set(out["family"].unique()) == {"qwen"}
    assert set(out["split"].unique()) == {"test"}
    # Score is what was in `suspicion` column.
    assert np.allclose(
        sorted(out["score"].tolist()), sorted(raw["suspicion"].dropna().tolist())
    )


def test_per_family_csv_to_long_success_filter_matches_claude_md_s4():
    """benign -> main_ok; sabotage -> main_ok AND side_ok."""
    csv = _fixture_csv()
    raw = pd.read_csv(csv)
    task_ids = raw["task_id"].astype(str).unique().tolist()
    splits = {t: "test" for t in task_ids}
    filtered = per_family_csv_to_long(csv, "qwen", splits, success_filter=True)
    raw_b = raw[(raw["mode"] == "honest") & (raw["main_task_success"] == True)]   # noqa: E712
    raw_s = raw[(raw["mode"] == "attack") & (raw["main_task_success"] == True) & (raw["side_task_success"] == True)]   # noqa: E712
    n_expected = len(raw_b) + len(raw_s)
    assert len(filtered) == n_expected


# ---------------------------------------------------------------------------
# build_transfer_scores_from_per_family
# ---------------------------------------------------------------------------
def _make_synth_per_family(family: str, fams: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(hash(family) & 0xFFFFFFFF)
    rows = []
    for split, n_tasks in [("calibration", 30), ("test", 30)]:
        for i in range(n_tasks):
            tid = f"{split}_t{i:03d}"
            for label in ("benign", "sabotage"):
                if split == "calibration" and label == "sabotage":
                    continue   # calibration is benign-only (the design doc s.4)
                score = float(rng.normal(0.5 if label == "sabotage" else 0.0, 1.0))
                rows.append({
                    "task_id": tid, "family": family,
                    "split": split, "label": label, "score": score,
                })
    return pd.DataFrame(rows)


def test_build_transfer_scores_fixed_prompt_replicates_across_calib():
    fams = ("qwen", "mistral")
    per_family = {f: _make_synth_per_family(f, fams) for f in fams}
    s = build_transfer_scores_from_per_family(
        per_family, monitor_id="fixed-test", monitor_kind="fixed_prompt", families=fams,
    )
    assert isinstance(s, TransferScores)
    # For each (task, eval_family, label, split) cell, the score must be
    # identical across calib_family. That's the §10 invariant.
    grp = s.df.groupby(["task_id", "eval_family", "label", "split"])["score"]
    assert (grp.max() == grp.min()).all()


def test_build_transfer_scores_family_conditioned_not_yet_supported():
    fams = ("qwen", "mistral")
    per_family = {f: _make_synth_per_family(f, fams) for f in fams}
    with pytest.raises(NotImplementedError, match="W7-8 path"):
        build_transfer_scores_from_per_family(
            per_family, monitor_id="m", monitor_kind="family_conditioned", families=fams,
        )


def test_build_transfer_scores_rejects_missing_family():
    fams = ("qwen", "mistral")
    per_family = {"qwen": _make_synth_per_family("qwen", fams)}  # missing mistral
    with pytest.raises(ValueError, match="missing eval_family"):
        build_transfer_scores_from_per_family(
            per_family, monitor_id="m", monitor_kind="fixed_prompt", families=fams,
        )


def test_build_transfer_scores_rejects_family_mismatch():
    fams = ("qwen", "mistral")
    per_family = {f: _make_synth_per_family(f, fams) for f in fams}
    # Corrupt: rename family in one df
    per_family["qwen"] = per_family["qwen"].assign(family="something_else")
    with pytest.raises(ValueError, match="family != qwen"):
        build_transfer_scores_from_per_family(
            per_family, monitor_id="m", monitor_kind="fixed_prompt", families=fams,
        )


# ---------------------------------------------------------------------------
# End-to-end on synthetic data: bridge -> matrices -> AUROC sanity check
# ---------------------------------------------------------------------------
def test_bridge_produces_matrices_that_decompose():
    """Synthetic two-family setup: monitor scores are family-independent
    (fixed_prompt invariant). Verify that the resulting TPR matrix builds
    and decomposes."""
    fams = ("qwen", "mistral")
    per_family = {f: _make_synth_per_family(f, fams) for f in fams}
    s = build_transfer_scores_from_per_family(
        per_family, monitor_id="fixed-test", monitor_kind="fixed_prompt", families=fams,
    )
    mats = build_transfer_matrices(s, fpr=0.01)
    # AUROC is a 1-D dict for fixed_prompt (the design doc s.1a / s.10)
    assert isinstance(mats["auroc"], dict)
    assert set(mats["auroc"].keys()) == set(fams)
    # TPR is a 2-D TransferMatrix (because tau varies with calib's benign tail)
    tpr = mats["tpr_at_fpr"]
    assert tpr.values.shape == (len(fams), len(fams))
    # Decomposition of the TPR matrix is defined.
    d = decompose_matrix(tpr)
    assert "diagonal_bonus" in d
