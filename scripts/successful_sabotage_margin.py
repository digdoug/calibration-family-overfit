"""(a) Ceiling-robustness check for the both-success interaction ($0, cached).

AUROC saturates near 1.0; when both diagonal and off-diagonal cells separate well
(diag 0.99-1.00, off 0.96 on the both-success cut) the *difference* is mechanically
compressed. So a small AUROC interaction (+0.029) could be a genuine effect OR a
ceiling artifact. This re-measures the SAME frontier-subset cells with a
NON-saturating separation metric -- Cohen's d on the row head's scores (mean
attack score - mean benign score, in pooled-SD units; unbounded) -- and compares
the attenuation RATIO (both-interaction / attempts-interaction) under each metric.

  ratio preserved under margin  -> attenuation is REAL (don't pay; small effect)
  ratio jumps toward 1 under margin -> AUROC was ceiling-limited (paying may reveal more)

Heads are fit once per family (cal split, basis row-all, PCA-50) and reused across
filters and metrics -- identical to the strict sub-block logic.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from track2_success_slice import keep_mask, load_with_outcomes
from track2_matrix_strict import diag_interaction, fit_row_model, short_name, transform_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def auroc(scores, labels):
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def cohen_d(scores, labels):
    a = scores[labels == 1]
    b = scores[labels == 0]
    if len(a) < 2 or len(b) < 2:
        return None
    psd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if psd < 1e-12:
        return None
    return float((a.mean() - b.mean()) / psd)


METRICS = {"auroc": auroc, "margin": cohen_d}


def matrix(fams, sub, heads, keep, metric, task_draw=None):
    cells = {}
    for a in sub:
        for b in sub:
            d = fams[b]
            base = (d["split"] == "test") & keep[b]
            if task_draw is None:
                idx = np.flatnonzero(base)
            else:
                chunks = [np.flatnonzero(base & (d["task"] == t)) for t in task_draw]
                idx = np.concatenate([c for c in chunks if len(c)]) if chunks else np.array([], int)
            if len(idx) == 0:
                return None
            v = METRICS[metric](transform_score(heads[a], d["emb"][idx]), d["label"][idx])
            if v is None:
                return None
            cells[(a, b)] = v
    return cells


def summ(cells, sub):
    diag = float(np.mean([cells[(x, x)] for x in sub]))
    off = float(np.mean([cells[(a, b)] for a in sub for b in sub if a != b]))
    return diag, off, diag_interaction(cells, sub)


def boot(fams, sub, heads, keep, metric, nb, seed):
    rng = np.random.default_rng(seed)
    tt = sorted({str(t) for d in fams.values() for t, sp in zip(d["task"], d["split"]) if sp == "test"})
    ints = []
    for _ in range(nb):
        draw = rng.choice(tt, size=len(tt), replace=True).tolist()
        c = matrix(fams, sub, heads, keep, metric, draw)
        if c is not None:
            ints.append(summ(c, sub)[2])
    a = np.array(ints)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), len(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", default="track2_n600_9fam")
    ap.add_argument("--families", default="gpt5,deepseek,grok,sonnet")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names, fams = load_with_outcomes(os.path.join(REPO, "results", args.roster))
    by_short = {short_name(n): n for n in names}
    sub = [by_short[w.strip()] for w in args.families.split(",") if w.strip()]
    heads = {a: fit_row_model(fams, a, "row-all", 50) for a in sub}

    print(f"Ceiling-robustness check -- families: {', '.join(short_name(n) for n in sub)}\n")
    base = {}
    for metric in ("auroc", "margin"):
        print(f"===== metric = {metric} ({'saturates at 1.0' if metric=='auroc' else 'unbounded effect size'}) =====")
        print(f"{'filter':>6} | {'diag':>7} {'off':>7} {'interaction':>10} {'95% CI':>20} {'draws':>6}")
        none_int = None
        for filt in ("none", "main", "both"):
            keep = {b: keep_mask(fams[b], filt) for b in sub}
            cells = matrix(fams, sub, heads, keep, metric)
            diag, off, inter = summ(cells, sub)
            im, lo, hi, nb = boot(fams, sub, heads, keep, metric, args.boot, args.seed)
            if filt == "none":
                none_int = inter
            ratio = f"  both/attempts={inter/none_int:+.2f}" if filt == "both" and none_int else ""
            print(f"{filt:>6} | {diag:>7.3f} {off:>7.3f} {inter:>+10.4f}  [{lo:>+.4f},{hi:>+.4f}] {nb:>5}{ratio}")
            base[(metric, filt)] = inter
        print()

    ra = base[("auroc", "both")] / base[("auroc", "none")]
    rm = base[("margin", "both")] / base[("margin", "none")]
    print("===== verdict =====")
    print(f"attenuation ratio (both / attempts):  AUROC {ra:+.2f}   margin {rm:+.2f}")
    if rm > ra + 0.25:
        print("-> margin ratio materially HIGHER: AUROC was ceiling-limited; realized-sabotage")
        print("   effect is larger than the AUROC interaction suggests (paying may reveal more).")
    else:
        print("-> ratios comparable: the attenuation is REAL, not a ceiling artifact.")
        print("   The realized-sabotage effect is genuinely smaller than the attempt-level one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
