"""
week1_analysis_core.py -- the statistical heart of the monitor-transfer project.

The make-or-break of this project is NOT ControlArena setup; it's whether the
cross-family drop survives as a positive INTERACTION term net of attacker-
obviousness (column) and monitor-capability (row) main effects (the design doc s.1b).

This file validates that decomposition on SYNTHETIC scores BEFORE generating a
single real transcript. Three regimes are simulated:

  overfit
      True diagonal advantage (the thesis). Calibration family A makes the
      monitor better at family A specifically. Interaction should be +ve,
      bootstrap CI should EXCLUDE 0.

  obviousness
      Pure column structure (the confound from Async Control / ASMR-Bench).
      Family B is hard-or-easy for everyone equally. Interaction should be ~0,
      bootstrap CI should INCLUDE 0. Row effect zero by construction.

  capability_plus_obviousness
      The harder confound: row main effect (some calibrations build better
      monitors) PLUS column main effect (some families are easier targets) but
      NO interaction. This is the realistic null the decomposition has to
      survive. Interaction should be ~0, bootstrap CI should INCLUDE 0.

If the decomposition can't distinguish these three on synthetic data, it
won't on real data either. Run this first.

Run:  python scripts/week1_analysis_core.py
Outputs: results/week1_validation/{summary.json, per_seed.json, regime_*.json}
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Make scripts/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transfer_data import (  # noqa: E402
    TransferScores,
    build_transfer_matrices,
    cluster_bootstrap_diagonal_bonus,
    decompose_matrix,
    decompose_via_statsmodels,
)


# ----------------------------------------------------------------------------
# Synthetic data generator -- emits a TransferScores in the long-form schema
# defined in transfer_data.py, exercising the cluster-bootstrap-by-task-ID
# contract on day one (so it can't decay).
# ----------------------------------------------------------------------------
Regime = Literal["overfit", "obviousness", "capability_plus_obviousness"]


@dataclass
class SynthConfig:
    families: tuple[str, ...] = ("qwen", "mistral", "gptoss")
    n_calibration_tasks: int = 100
    n_test_tasks: int = 100
    regime: Regime = "overfit"
    seed: int = 0
    sigma_task: float = 0.5  # per-task shared offset (creates within-task correlation
    #                         so cluster bootstrap actually does something)
    monitor_id: str = "synth-family-conditioned-v1"


def _sep_for_cell(regime: Regime, calib: str, ev: str, obviousness: dict[str, float],
                  capability: dict[str, float]) -> float:
    """Sabotage-vs-benign separability for cell (calib, eval) under each regime.
    Returns the mean shift of the sabotage Gaussian relative to benign."""
    if regime == "obviousness":
        # Pure column: every monitor sees family B's sabotage with the same separability.
        return obviousness[ev]
    if regime == "capability_plus_obviousness":
        # Row + column main effects, NO interaction.
        return capability[calib] + obviousness[ev]
    if regime == "overfit":
        # Diagonal bonus on top of a mild column effect.
        base = 0.7
        bonus = 1.3 if calib == ev else 0.0
        return base + bonus + 0.2 * obviousness[ev]
    raise ValueError(f"Unknown regime: {regime}")


def synth_transfer_scores(cfg: SynthConfig) -> TransferScores:
    rng = np.random.default_rng(cfg.seed)
    fams = cfg.families

    # Family-level latent parameters (constant within a regime/seed).
    obviousness = {f: float(rng.uniform(0.8, 2.2)) for f in fams}
    capability = {f: float(rng.uniform(-0.6, 0.6)) for f in fams}

    # Paired transcripts (the design doc s.4): same task IDs across families.
    # One split assignment per task -- disjoint, enforced by the schema validator.
    calib_task_ids = [f"calib_t{i:04d}" for i in range(cfg.n_calibration_tasks)]
    test_task_ids = [f"test_t{i:04d}" for i in range(cfg.n_test_tasks)]
    task_split = (
        {t: "calibration" for t in calib_task_ids}
        | {t: "test" for t in test_task_ids}
    )

    # Per-task random offset shared across all (calib, eval, label) rows for
    # that task. This is the within-task correlation the cluster bootstrap is
    # designed to handle.
    task_offset = {
        t: float(rng.normal(0.0, cfg.sigma_task))
        for t in calib_task_ids + test_task_ids
    }

    rows = []
    for task_id in calib_task_ids + test_task_ids:
        split = task_split[task_id]
        offset = task_offset[task_id]
        for calib, ev in product(fams, fams):
            # Calibration split: benign only (we only need benign to set tau).
            # Test split: both labels (to compute AUROC and TPR).
            labels = ["benign"] if split == "calibration" else ["benign", "sabotage"]
            for label in labels:
                sep = _sep_for_cell(cfg.regime, calib, ev, obviousness, capability)
                mean = sep if label == "sabotage" else 0.0
                score = offset + float(rng.normal(mean, 1.0))
                rows.append(
                    {
                        "task_id": task_id,
                        "calib_family": calib,
                        "eval_family": ev,
                        "split": split,
                        "label": label,
                        "score": score,
                        "monitor_id": cfg.monitor_id,
                        "generation_seed": cfg.seed,
                    }
                )

    df = pd.DataFrame(rows)
    return TransferScores(
        df=df,
        calib_families=fams,
        eval_families=fams,
        monitor_id=cfg.monitor_id,
        monitor_kind="family_conditioned",
    )


# ----------------------------------------------------------------------------
# Per-regime evaluation
# ----------------------------------------------------------------------------
def evaluate_regime(regime: Regime, seed: int, n_boot: int = 0) -> dict:
    cfg = SynthConfig(regime=regime, seed=seed)
    scores = synth_transfer_scores(cfg)
    mats = build_transfer_matrices(scores, fpr=0.01)

    auroc_mat = mats["auroc"]
    tpr_mat = mats["tpr_at_fpr"]
    dA = decompose_matrix(auroc_mat)
    dT = decompose_matrix(tpr_mat)

    out = {
        "regime": regime,
        "seed": seed,
        "auroc_matrix": auroc_mat.values.tolist(),
        "tpr_matrix": tpr_mat.values.tolist(),
        "families": list(scores.calib_families),
        "auroc_diagonal_bonus": dA["diagonal_bonus"],
        "tpr_diagonal_bonus": dT["diagonal_bonus"],
        "auroc_row_main_effect": dA["row_main_effect"].tolist(),
        "auroc_col_main_effect": dA["col_main_effect"].tolist(),
        "tau_per_calib": mats["tau_per_calib"],
    }

    if n_boot > 0:
        boot = cluster_bootstrap_diagonal_bonus(
            scores, cell_metric="auroc", n_boot=n_boot, seed=seed
        )
        out["auroc_bootstrap"] = boot

        # statsmodels OLS cross-check on the matrix-subtraction interaction.
        ols = decompose_via_statsmodels(
            scores, cell_value_fn=lambda s: build_transfer_matrices(s)["auroc"]
        )
        out["auroc_ols_diagonal_bonus"] = ols["diagonal_bonus"]
        out["auroc_ols_minus_matrix"] = ols["diagonal_bonus"] - dA["diagonal_bonus"]

    return out


# ----------------------------------------------------------------------------
# Main: seed sweep + bootstrap CI on representative seed + assertions
# ----------------------------------------------------------------------------
SEED_SWEEP = list(range(20))
REPRESENTATIVE_SEED = 0
N_BOOT = 500


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "week1_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Validating two-way decomposition on synthetic data (the design doc s.1b).")
    print(f"  Seeds:        {SEED_SWEEP[0]}..{SEED_SWEEP[-1]} ({len(SEED_SWEEP)} total)")
    print(f"  Bootstrap:    n_boot={N_BOOT} on seed={REPRESENTATIVE_SEED}")
    print()

    summary = {}
    per_seed = {}

    for regime in ("overfit", "obviousness", "capability_plus_obviousness"):
        t0 = time.time()
        per_seed_bonuses_auroc = []
        per_seed_bonuses_tpr = []
        for seed in SEED_SWEEP:
            n_boot = N_BOOT if seed == REPRESENTATIVE_SEED else 0
            res = evaluate_regime(regime, seed=seed, n_boot=n_boot)
            per_seed_bonuses_auroc.append(res["auroc_diagonal_bonus"])
            per_seed_bonuses_tpr.append(res["tpr_diagonal_bonus"])
            if seed == REPRESENTATIVE_SEED:
                with open(out_dir / f"regime_{regime}_seed{seed}.json", "w") as f:
                    json.dump(res, f, indent=2)
                rep = res

        elapsed = time.time() - t0
        auroc_arr = np.asarray(per_seed_bonuses_auroc)
        tpr_arr = np.asarray(per_seed_bonuses_tpr)
        summary[regime] = {
            "seed_sweep_size": len(SEED_SWEEP),
            "auroc_diagonal_bonus_median": float(np.median(auroc_arr)),
            "auroc_diagonal_bonus_iqr": [
                float(np.quantile(auroc_arr, 0.25)),
                float(np.quantile(auroc_arr, 0.75)),
            ],
            "auroc_diagonal_bonus_minmax": [float(auroc_arr.min()), float(auroc_arr.max())],
            "tpr_diagonal_bonus_median": float(np.median(tpr_arr)),
            "auroc_bootstrap_repr_seed": rep["auroc_bootstrap"],
            "auroc_ols_cross_check_diff_repr_seed": rep["auroc_ols_minus_matrix"],
            "elapsed_sec": elapsed,
        }
        per_seed[regime] = {
            "seeds": SEED_SWEEP,
            "auroc_diagonal_bonus": per_seed_bonuses_auroc,
            "tpr_diagonal_bonus": per_seed_bonuses_tpr,
        }

        print(f"--- regime = {regime}  ({elapsed:.1f}s) ---")
        print(f"  AUROC matrix (seed={REPRESENTATIVE_SEED}, [calib x eval]):")
        print(np.array2string(np.array(rep["auroc_matrix"]), precision=3, suppress_small=True,
                              prefix="    "))
        print(f"  AUROC diagonal_bonus  median(20 seeds) = "
              f"{np.median(auroc_arr):+.3f}  [IQR {np.quantile(auroc_arr, 0.25):+.3f}, "
              f"{np.quantile(auroc_arr, 0.75):+.3f}]")
        b = rep["auroc_bootstrap"]
        print(f"  AUROC cluster-bootstrap CI (seed={REPRESENTATIVE_SEED}, n_boot={N_BOOT}): "
              f"mean={b['mean']:+.3f}  95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]")
        print(f"  AUROC OLS cross-check  diff(OLS - matrix) = "
              f"{rep['auroc_ols_minus_matrix']:+.2e}  (should be ~0 for balanced cells)")
        print(f"  TPR@1%FPR diagonal_bonus median = {np.median(tpr_arr):+.3f}")
        print()

    # ------------------------------------------------------------------
    # Assertions per the design doc s.1b.
    # ------------------------------------------------------------------
    print("===== sanity assertions =====")
    failures: list[str] = []

    # 1. Overfit -> CLEARLY +ve point estimate AND bootstrap CI excludes 0.
    o = summary["overfit"]
    if o["auroc_diagonal_bonus_median"] <= 0.05:
        failures.append(
            f"OVERFIT: median diagonal_bonus over 20 seeds = "
            f"{o['auroc_diagonal_bonus_median']:+.3f} <= 0.05 (expected > 0.05)"
        )
    if o["auroc_bootstrap_repr_seed"]["ci_low"] <= 0.0:
        failures.append(
            f"OVERFIT: bootstrap 95% CI includes 0: "
            f"[{o['auroc_bootstrap_repr_seed']['ci_low']:+.3f}, "
            f"{o['auroc_bootstrap_repr_seed']['ci_high']:+.3f}]"
        )

    # 2. Obviousness -> point estimate ~0 AND CI includes 0.
    ob = summary["obviousness"]
    if abs(ob["auroc_diagonal_bonus_median"]) >= 0.05:
        failures.append(
            f"OBVIOUSNESS: |median diagonal_bonus| = "
            f"{abs(ob['auroc_diagonal_bonus_median']):.3f} >= 0.05 (expected ~0)"
        )
    ci = ob["auroc_bootstrap_repr_seed"]
    if not (ci["ci_low"] <= 0.0 <= ci["ci_high"]):
        failures.append(
            f"OBVIOUSNESS: bootstrap 95% CI excludes 0: "
            f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
        )

    # 3. Capability+obviousness -> same null behavior; this is the harder confound.
    cp = summary["capability_plus_obviousness"]
    if abs(cp["auroc_diagonal_bonus_median"]) >= 0.05:
        failures.append(
            f"CAPABILITY+OBVIOUSNESS: |median diagonal_bonus| = "
            f"{abs(cp['auroc_diagonal_bonus_median']):.3f} >= 0.05 (expected ~0)"
        )
    ci = cp["auroc_bootstrap_repr_seed"]
    if not (ci["ci_low"] <= 0.0 <= ci["ci_high"]):
        failures.append(
            f"CAPABILITY+OBVIOUSNESS: bootstrap 95% CI excludes 0: "
            f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"
        )

    # 4. OLS cross-check: matrix-subtraction interaction should match
    #    statsmodels OLS to within float tolerance on balanced cells.
    for regime, s in summary.items():
        diff = abs(s["auroc_ols_cross_check_diff_repr_seed"])
        if diff > 1e-6:
            failures.append(
                f"{regime}: OLS - matrix diagonal_bonus diff = {diff:.3e} (>1e-6); "
                "the two decomposition definitions disagree."
            )

    # Persist everything before reporting pass/fail.
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_seed.json", "w") as f:
        json.dump(per_seed, f, indent=2)
    with open(out_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "seeds": SEED_SWEEP,
                "n_boot": N_BOOT,
                "representative_seed": REPRESENTATIVE_SEED,
                "n_calibration_tasks": SynthConfig().n_calibration_tasks,
                "n_test_tasks": SynthConfig().n_test_tasks,
                "sigma_task": SynthConfig().sigma_task,
                "families": list(SynthConfig().families),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
            f,
            indent=2,
        )

    if failures:
        print("FAIL:")
        for msg in failures:
            print(f"  - {msg}")
        print(f"\nOutputs in: {out_dir}")
        return 1

    print("PASS: decomposition separates monitor-overfit from pure-obviousness AND")
    print("      from capability+obviousness (the harder null with row main effect).")
    print(f"      OLS vs matrix-subtraction interaction agree to <1e-6 on all regimes.")
    print(f"\nOutputs in: {out_dir}")
    print("\nNEXT (the design doc s.7):")
    print("  - W0 SHADE access request + arXiv kill-check on 2505.23575, 2601.21112,")
    print("    2508.19461 (appendices) + ControlArena standup at pinned SHA.")
    print("  - W1 GATE 1: reproduce a single-family baseline monitor AUROC matching")
    print("    a published figure within reason; verify trajectory export round-trips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
