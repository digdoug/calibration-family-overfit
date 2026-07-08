"""Multiplicity-preserving task-cluster bootstrap helper.

THE BUG (fixed by using this helper): the original track2 cluster bootstraps
resampled TEST tasks with replacement, then collapsed the draw with
``set(...)`` and a boolean membership mask ``[t in keep for t in task]``::

    keep = set(rng.choice(test_tasks, size=len(test_tasks), replace=True).tolist())
    m = (split == "test") & np.array([t in keep for t in task])

``set()`` discards MULTIPLICITY: a task drawn k>1 times contributes its rows
exactly ONCE instead of k times. The resample is then really a random *subset*
of tasks (each task included w.p. 1-(1-1/n)^n ~= 0.632), NOT a true bootstrap.
That understates between-task (cluster) variance, so the resulting 95% CIs are
too NARROW. The adopted strict scripts (track2_matrix_strict,
track2_safety_strict, track2_success_slice) already do this correctly by
concatenating per-drawn-task row chunks; this helper lets the remaining control
scripts use the identical, correct estimator.

Usage (drop-in for the ``keep = set(...)`` + membership-mask pattern)::

    rows = {B: task_rows(fams[B]["task"], eligible_B, test_tasks) for B in names}
    for _ in range(n_boot):
        draw = rng.choice(test_tasks, size=len(test_tasks), replace=True)
        idxs = {B: draw_index(rows[B], draw) for B in names}
        ...  # use idxs[B] where the old code used the boolean mask m on family B
"""
from __future__ import annotations

import numpy as np


def task_rows(task: np.ndarray, eligible: np.ndarray, tasks_universe) -> dict:
    """Precompute, once, a map task -> array of ELIGIBLE row indices for one family.

    ``eligible`` is the boolean mask the old code ANDed in before the membership
    test (e.g. ``split == "test"`` optionally also ``& d["keep"]``). Tasks in
    ``tasks_universe`` absent from this family map to an empty array.
    """
    rows = {}
    for t in tasks_universe:
        rows[t] = np.flatnonzero(eligible & (task == t))
    return rows


def draw_index(rows: dict, draw) -> np.ndarray:
    """Expand a resampled task draw into a row-index array, preserving multiplicity.

    A task appearing k times in ``draw`` contributes its rows k times -- the true
    cluster bootstrap. Returns an int index array (possibly empty).
    """
    parts = [rows[t] for t in draw if t in rows and len(rows[t])]
    if not parts:
        return np.empty(0, dtype=int)
    return np.concatenate(parts)
