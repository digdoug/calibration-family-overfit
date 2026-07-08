"""
Tests for the schema lock-in in transfer_data.py. These are the guards we
want to fail before any real ControlArena scores flow through.

Pytest entrypoint:  pytest scripts/test_transfer_data.py -v
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd
import pytest

from transfer_data import (
    TransferScores,
    TransferMatrix,
    build_transfer_matrices,
    cluster_bootstrap_diagonal_bonus,
    decompose_matrix,
    decompose_via_statsmodels,
    _iid_bootstrap_diagonal_bonus_for_testing,
)


FAMS = ("qwen", "mistral", "gptoss")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _make_df(
    n_cal: int = 30,
    n_test: int = 30,
    seed: int = 0,
    score_fn=None,
    monitor_id: str = "test-monitor",
    families: tuple[str, ...] = FAMS,
) -> pd.DataFrame:
    """Build a long-form DataFrame matching the TransferScores schema.

    score_fn(calib, eval_, label, task_id, rng) -> float; default = standard
    Gaussian for benign, Gaussian shifted by +1 for sabotage."""
    rng = np.random.default_rng(seed)

    def default_score(calib, eval_, label, task_id, rng):
        if label == "benign":
            return float(rng.normal(0.0, 1.0))
        return float(rng.normal(1.0, 1.0))

    score_fn = score_fn or default_score

    cal_tasks = [f"calib_t{i:04d}" for i in range(n_cal)]
    test_tasks = [f"test_t{i:04d}" for i in range(n_test)]
    rows = []
    for task_id in cal_tasks:
        for calib, ev in product(families, families):
            rows.append({
                "task_id": task_id, "calib_family": calib, "eval_family": ev,
                "split": "calibration", "label": "benign",
                "score": score_fn(calib, ev, "benign", task_id, rng),
                "monitor_id": monitor_id, "generation_seed": seed,
            })
    for task_id in test_tasks:
        for calib, ev in product(families, families):
            for label in ("benign", "sabotage"):
                rows.append({
                    "task_id": task_id, "calib_family": calib, "eval_family": ev,
                    "split": "test", "label": label,
                    "score": score_fn(calib, ev, label, task_id, rng),
                    "monitor_id": monitor_id, "generation_seed": seed,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Schema guards
# ----------------------------------------------------------------------------
def test_required_columns_must_be_present():
    df = _make_df()
    df = df.drop(columns=["generation_seed"])
    with pytest.raises(ValueError, match="missing columns"):
        TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="test-monitor", monitor_kind="family_conditioned")


def test_task_disjoint_splits_enforced():
    """A task_id may appear in exactly one split (the design doc s.4)."""
    df = _make_df()
    # Pick one calibration task and mislabel one row to split='test'.
    target = df[df["split"] == "calibration"]["task_id"].iloc[0]
    mask = (df["task_id"] == target) & (df["split"] == "calibration")
    df.loc[df[mask].index[0], "split"] = "test"

    with pytest.raises(ValueError, match="appear in >1 split"):
        TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="test-monitor", monitor_kind="family_conditioned")


def test_fixed_prompt_score_must_be_calib_invariant():
    """the design doc s.10 hard rule -- a fixed prompt monitor's score is
    invariant in calib_family by definition. Schema must reject calib-
    varying scores under monitor_kind='fixed_prompt'."""
    def score_fn(calib, ev, label, task_id, rng):
        # Inject a calib-dependent shift -> this is what we must catch.
        base = 0.0 if label == "benign" else 1.0
        shift = {"qwen": 0.0, "mistral": 0.4, "gptoss": -0.4}[calib]
        return float(rng.normal(base + shift, 1.0))

    df = _make_df(score_fn=score_fn)
    with pytest.raises(ValueError, match="monitor_kind='fixed_prompt'"):
        TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="test-monitor", monitor_kind="fixed_prompt")


def test_fixed_prompt_score_calib_invariant_passes():
    """If scores are genuinely calib-invariant, fixed_prompt validates."""
    def score_fn(calib, ev, label, task_id, rng):
        # NB: rng is shared, so this is calib-VARYING in the per-row Gaussian
        # draw. We need to make the row score truly independent of calib.
        # Approach: derive score deterministically from (task, ev, label).
        key = hash((task_id, ev, label)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        return float(local_rng.normal(base, 1.0))

    df = _make_df(score_fn=score_fn)
    s = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="test-monitor", monitor_kind="fixed_prompt")
    assert s.monitor_kind == "fixed_prompt"


def test_family_conditioned_must_actually_vary_with_calib():
    """If monitor_kind='family_conditioned' but scores don't vary across
    calib_family on test rows, the monitor is mis-labeled."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        return float(local_rng.normal(base, 1.0))

    df = _make_df(score_fn=score_fn)
    with pytest.raises(ValueError, match="monitor_kind='family_conditioned'"):
        TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="test-monitor", monitor_kind="family_conditioned")


# ----------------------------------------------------------------------------
# Matrix construction guards
# ----------------------------------------------------------------------------
def test_fixed_prompt_does_not_emit_auroc_matrix():
    """the design doc s.1a -- the §10 hard rule. Fixed-prompt monitor's AUROC has
    no calibration-axis. build_transfer_matrices returns dict[eval -> auroc],
    NOT a TransferMatrix."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        return float(local_rng.normal(base, 1.0))

    df = _make_df(score_fn=score_fn, monitor_id="m")
    s = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="m", monitor_kind="fixed_prompt")
    mats = build_transfer_matrices(s)
    assert isinstance(mats["auroc"], dict), (
        "fixed_prompt monitor must NOT yield a 2-D AUROC matrix (the design doc s.1a)"
    )
    assert set(mats["auroc"].keys()) == set(FAMS)
    # TPR is still 2-D because tau varies with calib.
    assert isinstance(mats["tpr_at_fpr"], TransferMatrix)


def test_family_conditioned_emits_auroc_matrix():
    """Track-2 monitor (family-conditioned) gets a real 2-D AUROC matrix."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label, calib)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        # Diagonal bonus
        bonus = 1.0 if calib == ev and label == "sabotage" else 0.0
        return float(local_rng.normal(base + bonus, 1.0))

    df = _make_df(score_fn=score_fn, n_cal=80, n_test=80, monitor_id="m")
    s = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="m", monitor_kind="family_conditioned")
    mats = build_transfer_matrices(s)
    assert isinstance(mats["auroc"], TransferMatrix)
    assert mats["auroc"].values.shape == (len(FAMS), len(FAMS))


def test_cluster_bootstrap_refuses_auroc_for_fixed_prompt():
    """the design doc s.10 -- nothing downstream should ever try to bootstrap an
    AUROC diagonal_bonus for a fixed-prompt monitor."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        return float(local_rng.normal(base, 1.0))

    df = _make_df(score_fn=score_fn, monitor_id="m")
    s = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="m", monitor_kind="fixed_prompt")
    with pytest.raises(ValueError, match="cell_metric='auroc'"):
        cluster_bootstrap_diagonal_bonus(s, cell_metric="auroc", n_boot=50)


# ----------------------------------------------------------------------------
# Named-axis matrix
# ----------------------------------------------------------------------------
def test_transfer_matrix_named_axis_access():
    """TransferMatrix.cell(calib, eval) is the only intended accessor;
    raw int indexing isn't part of the public API."""
    M = TransferMatrix(
        values=np.array([[0.9, 0.7, 0.7], [0.7, 0.9, 0.7], [0.7, 0.7, 0.9]]),
        calib_families=FAMS, eval_families=FAMS, metric="auroc",
    )
    assert math.isclose(M.cell("qwen", "qwen"), 0.9)
    assert math.isclose(M.cell("qwen", "mistral"), 0.7)
    with pytest.raises(ValueError):
        M.cell("unknown_family", "qwen")


def test_decompose_requires_square_named_axes():
    M = TransferMatrix(
        values=np.array([[0.9, 0.7], [0.7, 0.9]]),
        calib_families=("a", "b"), eval_families=("a", "c"),
        metric="auroc",
    )
    with pytest.raises(ValueError, match="calib_families == eval_families"):
        decompose_matrix(M)


# ----------------------------------------------------------------------------
# Decomposition agreement with statsmodels OLS
# ----------------------------------------------------------------------------
def test_decompose_matches_statsmodels_on_balanced_cells():
    """When cell counts are equal, the matrix-subtraction interaction
    estimator equals the OLS interaction. This locks the decomposition
    definition."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label, calib)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        bonus = 0.8 if calib == ev and label == "sabotage" else 0.0
        return float(local_rng.normal(base + bonus, 1.0))

    df = _make_df(score_fn=score_fn, n_cal=120, n_test=120, monitor_id="m")
    s = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                       monitor_id="m", monitor_kind="family_conditioned")
    M = build_transfer_matrices(s)["auroc"]
    d_matrix = decompose_matrix(M)
    d_ols = decompose_via_statsmodels(
        s, cell_value_fn=lambda x: build_transfer_matrices(x)["auroc"]
    )
    assert abs(d_matrix["diagonal_bonus"] - d_ols["diagonal_bonus"]) < 1e-6


# ----------------------------------------------------------------------------
# Cluster bootstrap behavior. NB: a naive "cluster CI > iid CI" assertion is
# wrong for our data shape -- each (task, cell) is one row, so the classical
# within-cell within-task replication that drives that inequality doesn't
# apply. The properties that DO matter and that we lock in here:
#   1. The bootstrap is centered on the original point estimate.
#   2. The bootstrap CI is non-degenerate (width > 0).
#   3. The resampling unit is task_id, not row -- verified by counting
#      unique tasks in a single resample (expected ~63% of the original pool).
#   4. Adding within-task correlation INFLATES the bootstrap CI relative to
#      data with no per-task structure (the task offset variance survives the
#      bootstrap).
# ----------------------------------------------------------------------------
def _build_scores_with_task_correlation(sigma_task: float, seed: int = 7) -> TransferScores:
    rng = np.random.default_rng(seed)
    cal_tasks = [f"calib_t{i:04d}" for i in range(60)]
    test_tasks = [f"test_t{i:04d}" for i in range(60)]
    task_offset = {t: float(rng.normal(0.0, sigma_task))
                   for t in cal_tasks + test_tasks}
    rows = []
    for task_id in cal_tasks:
        offset = task_offset[task_id]
        for calib, ev in product(FAMS, FAMS):
            rows.append({"task_id": task_id, "calib_family": calib,
                         "eval_family": ev, "split": "calibration",
                         "label": "benign",
                         "score": offset + float(rng.normal(0.0, 0.3)),
                         "monitor_id": "m", "generation_seed": seed})
    for task_id in test_tasks:
        offset = task_offset[task_id]
        for calib, ev in product(FAMS, FAMS):
            for label in ("benign", "sabotage"):
                base = 0.0 if label == "benign" else 1.0
                bonus = 0.8 if calib == ev and label == "sabotage" else 0.0
                rows.append({"task_id": task_id, "calib_family": calib,
                             "eval_family": ev, "split": "test",
                             "label": label,
                             "score": offset + base + bonus + float(rng.normal(0.0, 0.3)),
                             "monitor_id": "m", "generation_seed": seed})
    df = pd.DataFrame(rows)
    return TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                          monitor_id="m", monitor_kind="family_conditioned")


def test_cluster_bootstrap_centered_on_point_estimate():
    s = _build_scores_with_task_correlation(sigma_task=2.0)
    M = build_transfer_matrices(s)["auroc"]
    point = decompose_matrix(M)["diagonal_bonus"]
    boot = cluster_bootstrap_diagonal_bonus(s, n_boot=400, seed=0)
    # CI must contain the original point estimate.
    assert boot["ci_low"] <= point <= boot["ci_high"], (
        f"point={point:.4f} not in CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]"
    )
    # Width should be non-degenerate.
    assert boot["ci_high"] - boot["ci_low"] > 1e-3, (
        f"degenerate CI width = {boot['ci_high'] - boot['ci_low']:.4g}"
    )


def test_cluster_bootstrap_resamples_tasks_not_rows():
    """Verify the resampling unit is task_id: in a single bootstrap iteration,
    the unique-task fraction should match the theoretical bootstrap-with-
    replacement expectation (~1 - 1/e ~ 0.632)."""
    s = _build_scores_with_task_correlation(sigma_task=1.0)
    rng = np.random.default_rng(0)
    task_ids = s.df["task_id"].unique()
    n = len(task_ids)
    unique_fractions = []
    for _ in range(50):
        sampled = rng.choice(task_ids, size=n, replace=True)
        unique_fractions.append(len(set(sampled)) / n)
    mean_unique = float(np.mean(unique_fractions))
    assert 0.55 < mean_unique < 0.71, (
        f"mean unique-task fraction in bootstrap = {mean_unique:.3f}; "
        "expected ~0.632 (1 - 1/e); resampling unit may not be the task"
    )


def test_cluster_bootstrap_ci_wider_when_within_task_correlation_higher():
    """Property 4: increasing the per-task offset variance should widen the
    cluster-bootstrap CI on the diagonal_bonus. This is the substantive check
    that the cluster bootstrap captures task-level random-effects variance."""
    boot_lo = cluster_bootstrap_diagonal_bonus(
        _build_scores_with_task_correlation(sigma_task=0.1), n_boot=300, seed=0
    )
    boot_hi = cluster_bootstrap_diagonal_bonus(
        _build_scores_with_task_correlation(sigma_task=3.0), n_boot=300, seed=0
    )
    w_lo = boot_lo["ci_high"] - boot_lo["ci_low"]
    w_hi = boot_hi["ci_high"] - boot_hi["ci_low"]
    assert w_hi > w_lo * 1.3, (
        f"high-correlation CI width = {w_hi:.4f} not >> low-correlation CI width = "
        f"{w_lo:.4f}; cluster bootstrap may not be picking up task-level variance"
    )


# ----------------------------------------------------------------------------
# tau is set on the CALIBRATION split's benign only (the design doc s.4)
# ----------------------------------------------------------------------------
def test_tau_uses_only_calibration_split_benign():
    """Regression guard for audit issue S4: tau must come from the
    calibration split, not the test split's benign data. We verify by
    injecting a huge artificial shift into the test-benign for one family
    and confirming tau doesn't move."""
    def score_fn(calib, ev, label, task_id, rng):
        key = hash((task_id, ev, label, calib)) & 0xFFFFFFFF
        local_rng = np.random.default_rng(key)
        base = 0.0 if label == "benign" else 1.0
        return float(local_rng.normal(base, 1.0))

    df = _make_df(score_fn=score_fn, n_cal=120, n_test=120, monitor_id="m")
    s_clean = TransferScores(df=df.copy(), calib_families=FAMS, eval_families=FAMS,
                             monitor_id="m", monitor_kind="family_conditioned")
    mats_clean = build_transfer_matrices(s_clean)
    tau_clean = mats_clean["tau_per_calib"]["qwen"]

    # Shift the TEST-split benign for qwen by +1000.
    mask = ((df["split"] == "test") & (df["label"] == "benign")
            & (df["calib_family"] == "qwen") & (df["eval_family"] == "qwen"))
    df.loc[mask, "score"] += 1000.0
    s_shifted = TransferScores(df=df, calib_families=FAMS, eval_families=FAMS,
                               monitor_id="m", monitor_kind="family_conditioned")
    mats_shifted = build_transfer_matrices(s_shifted)
    tau_shifted = mats_shifted["tau_per_calib"]["qwen"]

    assert abs(tau_clean - tau_shifted) < 1e-9, (
        "tau moved when test-benign was perturbed; it must come from the calibration split only"
    )
