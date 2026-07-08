"""Control 15: drop-one-family JACKKNIFE of the interaction term.

The deepest gate-free robustness objection a reviewer raises against the headline:
"your diagonal-dominant interaction is carried by ONE family (mistral, whose
diagonal sits at 0.96-1.00 in every cut); drop it and the effect collapses."
The controls repeatedly NOTE mistral's dominance (Controls 7/10/11/12/13) but
never resolve it. This does.

For each family f, REMOVE it from the matrix entirely (both as a calibration row
and an evaluation column), refit the WHOLE pipeline (pooled-cal StandardScaler +
PCA-50 + per-family logistic heads) on the remaining N-1 families, and recompute
the interaction + cluster-bootstrap CI by task. If the interaction stays CI>0
with the dominant family removed, the effect is NOT a single-family artifact.

Honest jackknife: refit everything on the reduced family set (not just slice the
matrix) so the pooled PCA subspace and the main-effect removal both reflect the
N-1 world — the actual "what if this lineage were not in the study" counterfactual.

Reuses track2_matrix's exact split / diag_interaction / cluster_boot so the
numbers are directly comparable to the headline.

Run: .venv/bin/python scripts/track2_jackknife.py [npz_dir=track2_n600]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track2_matrix import split_of, diag_interaction, cluster_boot  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dir(npz_dir):
    fams = {}
    for f in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        name = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"], "task": d["task_id"]}
    for d in fams.values():
        d["split"] = np.array([split_of(t) for t in d["task"]])
    return fams


def fit_and_score(fams, names):
    """Refit pooled scaler+PCA+per-family heads on exactly `names`; return interaction + boot CI."""
    sub = {n: fams[n] for n in names}
    cal_emb = np.vstack([d["emb"][d["split"] == "cal"] for d in sub.values()])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    for d in sub.values():
        d["p"] = pca.transform(scaler.transform(d["emb"]))
    heads = {}
    for A in names:
        d = sub[A]; cal = d["split"] == "cal"
        heads[A] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    M = {}
    for A in names:
        for B in names:
            d = sub[B]; te = d["split"] == "test"
            from sklearn.metrics import roc_auc_score
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    inter = diag_interaction(M, names)
    diag = np.mean([M[(x, x)] for x in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    rm, rlo, rhi, im, ilo, ihi, nb = cluster_boot(sub, names, heads)
    return dict(inter=inter, diag=diag, off=off, im=im, ilo=ilo, ihi=ihi, nb=nb)


def short(n):
    return n.replace("ow_", "").replace("_n600", "").replace("_n200", "").replace("frontier_", "").replace("_flash", "")


def main():
    npz_dir = os.path.join(REPO, "results", sys.argv[1] if len(sys.argv) > 1 else "track2_n600")
    fams = load_dir(npz_dir)
    names = list(fams)
    print(f"\n===== Drop-one-family JACKKNIFE of the interaction — {os.path.basename(npz_dir)} =====")
    print(f"families ({len(names)}): {', '.join(short(n) for n in names)}\n")

    full = fit_and_score(fams, names)
    print(f"FULL ({len(names)} fam): interaction = {full['inter']:+.4f}  "
          f"CI [{full['ilo']:+.4f},{full['ihi']:+.4f}]  (diag {full['diag']:.3f} / off {full['off']:.3f})\n")

    print(f"{'dropped':>10} | {'interaction':>11} | {'95% CI':>20} | {'diag':>5} {'off':>5} | Δinter vs full")
    print("-" * 78)
    rows = []
    for f in names:
        keep = [n for n in names if n != f]
        r = fit_and_score(fams, keep)
        d_inter = r["inter"] - full["inter"]
        flag = "" if r["ilo"] > 0 else "  <-- CI crosses 0!"
        print(f"{short(f):>10} | {r['inter']:>+11.4f} | [{r['ilo']:+.4f},{r['ihi']:+.4f}] | "
              f"{r['diag']:.3f} {r['off']:.3f} | {d_inter:+.4f}{flag}")
        rows.append((f, r, d_inter))

    surviving = [r for _, r, _ in rows if r["ilo"] > 0]
    print("\n" + "=" * 78)
    print(f"VERDICT: interaction CI>0 with EVERY single family removed: "
          f"{'YES' if len(surviving) == len(rows) else 'NO'} "
          f"({len(surviving)}/{len(rows)} drops keep CI>0).")
    worst = min(rows, key=lambda x: x[1]["inter"])
    print(f"Largest single-family contribution: dropping {short(worst[0])} -> "
          f"interaction {worst[1]['inter']:+.4f} (Δ {worst[2]:+.4f}); "
          f"min interaction across all drops still {'>0 (CI clear)' if worst[1]['ilo']>0 else 'CI touches 0'}.")


if __name__ == "__main__":
    raise SystemExit(main())
