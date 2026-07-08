"""Figure for the causal head-angle intervention (docs/figures/fig8_intervention.{pdf,png}).

Self-contained + additive: recomputes point estimates from the cached embeddings via
scripts/track2_intervention.py (no paper prose or existing figure touched). Three panels,
all on the four-open primary roster:

  A. Causal sweep: pooled recovery fraction vs rotation fraction f. Aligning head A onto B
     (slerp) recovers AUROC monotonically; an EQUAL-MAGNITUDE random rotation does not, and
     benign-only CORAL/mean-shift (Control 20) do not. Bands = across-pair spread.
  B. The unifying law: every intervened direction plotted as (achieved cosine to b_hat,
     transfer AUROC). Oracle-slerp and labeled-budget-probe points fall on ONE monotone
     curve; equal-magnitude random rotations sit off it at baseline cosine.
  C. Labeled-budget recovery curve: diff-of-means direction probe vs budget m, with the
     random-label control. Band = cluster-bootstrap 95% CI from the JSON if present, else
     across-family spread.

Run:  .venv/bin/python scripts/make_fig_intervention.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import track2_intervention as ti
import track2_matrix_strict as tms

REPO = ti.REPO
OUT = os.path.join(REPO, "docs", "figures")
ROSTER = "track2_n600"
BASIS = "row-all"
C_ALIGN, C_RAND, C_CORAL, C_PROBE = "#2166ac", "#b2182b", "#777777", "#1a9850"


def recover_curves(C, names):
    """Pooled-over-pairs recovery fraction at each f for aligned + random, plus benign points."""
    pairs = [(a, b) for a in names for b in names if a != b]
    F = ti.F_GRID
    al = {f: [] for f in F}
    rd = {f: [] for f in F}
    cor, ms = [], []
    for (a, b) in pairs:
        fl, ce = C["floor"][(a, b)], C["ceil"][(a, b)]
        if ce - fl <= ti.GAP_MIN:
            continue
        for f in F:
            al[f].append(ti.recovery(C["aligned_f"][(a, b)][f], fl, ce))
            rd[f].append(ti.recovery(C["random_f"][(a, b)][f], fl, ce))
        cor.append(ti.recovery(C["coral"][(a, b)], fl, ce))
        ms.append(ti.recovery(C["meanshift"][(a, b)], fl, ce))
    g = lambda dd: (np.array([np.nanmean(dd[f]) for f in F]),
                    np.array([np.nanstd(dd[f]) for f in F]))
    return g(al), g(rd), float(np.nanmean(cor)), float(np.nanmean(ms))


def budget_curve(C, names):
    """Pooled-over-B recovery fraction at each m for the probe + random-label control."""
    pr, rl = {m: [] for m in ti.M_GRID}, {m: [] for m in ti.M_GRID}
    for b in names:
        others = [a for a in names if a != b]
        fl = float(np.nanmean([C["floor"][(a, b)] for a in others]))
        ce = float(np.nanmean([C["ceil"][(a, b)] for a in others]))
        if ce - fl <= ti.GAP_MIN:
            continue
        for m in ti.M_GRID:
            if m in C["probe_m"].get(b, {}):
                pr[m].append(ti.recovery(C["probe_m"][b][m], fl, ce))
            if m in C["randlabel_m"].get(b, {}):
                rl[m].append(ti.recovery(C["randlabel_m"][b][m], fl, ce))
    ms = [m for m in ti.M_GRID if pr[m]]
    g = lambda dd: (np.array([np.nanmean(dd[m]) for m in ms]),
                    np.array([np.nanstd(dd[m]) for m in ms]))
    return ms, g(pr), g(rl)


def main():
    names, fams = tms.load_families(os.path.join(REPO, "results", ROSTER))
    names = sorted(names)
    _, _, per_b = ti.build(fams, names, BASIS, 50, 0)
    C = ti.cell_aurocs(per_b, names, which="FULL")
    (al_m, al_s), (rd_m, rd_s), cor, ms = recover_curves(C, names)
    mgrid, (pr_m, pr_s), (rl_m, rl_s) = budget_curve(C, names)
    ca = np.array(C["ca"]); cp = np.array(C["cp"]); cr = np.array(C["cr"])

    # optional bootstrap CI band for panel C from a saved JSON run
    boot_ci = None
    jpath = os.path.join(REPO, "results", "intervention", "4open.json")
    if os.path.exists(jpath):
        with open(jpath) as fh:
            jd = json.load(fh)
        rp = jd.get("bootstrap", {}).get("rec_probe_m", {})
        if rp:
            boot_ci = {int(k): v for k, v in rp.items()}

    F = np.array(ti.F_GRID)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # ---- Panel A: causal sweep ----
    ax = axes[0]
    ax.fill_between(F, al_m - al_s, al_m + al_s, color=C_ALIGN, alpha=0.15, zorder=1)
    ax.plot(F, al_m, "-o", color=C_ALIGN, lw=2, ms=5, zorder=4,
            label="align onto B (slerp)")
    ax.fill_between(F, rd_m - rd_s, rd_m + rd_s, color=C_RAND, alpha=0.12, zorder=1)
    ax.plot(F, rd_m, "-s", color=C_RAND, lw=2, ms=4, zorder=3,
            label="equal-magnitude random rotation")
    ax.axhline(cor, color=C_CORAL, ls="--", lw=1.5, zorder=2, label=f"benign CORAL ({cor:+.02f})")
    ax.axhline(0.0, color="k", ls=":", lw=1, zorder=1)
    ax.axhline(1.0, color="grey", ls=":", lw=1, zorder=1)
    ax.text(0.02, 1.0, " diagonal ceiling", color="grey", fontsize=7, va="bottom")
    ax.text(0.02, 0.0, " off-lineage floor", color="k", fontsize=7, va="bottom")
    ax.set_xlabel("rotation fraction $f$  (0 = head A,  1 = head B)")
    ax.set_ylabel("recovery fraction of the transfer gap")
    ax.set_title("A. Rotate the head onto B → AUROC recovers\n(random rotation of equal angle does not)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center left", frameon=True)
    ax.set_ylim(-0.25, 1.18)

    # ---- Panel B: the unifying law ----
    ax = axes[1]
    ax.scatter(cr[:, 0], cr[:, 1], s=10, color=C_RAND, alpha=0.35, edgecolor="none",
               zorder=2, label="random rotation (off the law)")
    ax.scatter(ca[:, 0], ca[:, 1], s=22, color=C_ALIGN, alpha=0.7, edgecolor="white",
               linewidth=0.3, zorder=3, label="oracle slerp")
    ax.scatter(cp[:, 0], cp[:, 1], s=14, color=C_PROBE, alpha=0.55, edgecolor="none",
               zorder=3, label="labeled-budget probe")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 0.5, " chance ", color="grey", fontsize=7, va="bottom", ha="right")
    ax.set_xlabel(r"achieved cosine to B's direction  $\cos(w,\,w_B)$")
    ax.set_ylabel("transfer AUROC on B")
    ax.set_title("B. One law: AUROC is governed by the achieved\ncosine, however the rotation is produced",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=True)

    # ---- Panel C: labeled-budget recovery curve ----
    ax = axes[2]
    x = np.array(mgrid)
    if boot_ci is not None:
        lo = np.array([boot_ci[m][1] for m in mgrid])
        hi = np.array([boot_ci[m][2] for m in mgrid])
        ax.fill_between(x, lo, hi, color=C_PROBE, alpha=0.18, zorder=1,
                        label="probe 95% CI (cluster boot)")
    else:
        ax.fill_between(x, pr_m - pr_s, pr_m + pr_s, color=C_PROBE, alpha=0.15, zorder=1)
    ax.plot(x, pr_m, "-o", color=C_PROBE, lw=2, ms=5, zorder=4,
            label="diff-of-means probe")
    ax.plot(x, rl_m, "-s", color=C_RAND, lw=1.8, ms=4, zorder=3,
            label="random-label control")
    ax.axhline(cor, color=C_CORAL, ls="--", lw=1.5, zorder=2, label=f"benign CORAL ({cor:+.02f})")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mgrid])
    ax.set_xlabel("labeled budget per class  $m$  (B sabotage + B benign)")
    ax.set_ylabel("recovery fraction of the transfer gap")
    ax.set_title("C. Labeled-budget recovery curve\n(a small labeled set rotates A most of the way)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left", frameon=True)
    ax.set_ylim(-0.25, 1.12)

    fig.suptitle("The head-angle mechanism is causal: rotating the calibration head onto the "
                 "evaluation family recovers transfer AUROC (four-open primary roster)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig8_intervention.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig8_intervention (aligned/random/CORAL sweep + unifying law + budget curve)")
    print(f"  panel A: aligned f=1 recovery {al_m[-1]:+.2f}, random {rd_m[-1]:+.2f}, CORAL {cor:+.2f}")
    print(f"  panel C: probe recovery m=1 {pr_m[0]:+.2f} -> m={mgrid[-1]} {pr_m[-1]:+.2f}")


if __name__ == "__main__":
    main()
