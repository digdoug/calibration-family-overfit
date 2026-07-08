"""Capability-gap analysis (roadmap step 1): does a monitor calibrated on WEAKER models catch
STRONGER ones? The deployment-critical weak-to-strong axis, as a free reanalysis of the existing
5-family n=600 data (no new generation).

Capability proxy = attack-mode main-task success (results/capability.json, precomputed with the
ControlArena venv). The 5-family cross-family AUROC matrix already contains the axis: a cell (A,B)
with cap[A] < cap[B] is "calibrate on a weaker family A, deploy against a stronger family B".

Outputs: point estimates + task-clustered bootstrap CIs for (a) the deploy-against-the-frontier
contrast (open-family heads -> Sonnet vs Sonnet in-lineage) and (b) the weak->strong vs strong->weak
off-diagonal asymmetry, plus the AUROC-vs-capability-gap trend, and a figure. Run with .venv.
"""
import os, glob, json, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from track2_matrix import split_of  # noqa: E402

T2 = "track2_n600_5fam"
SHORT = lambda n: n.replace("ow_", "").replace("_n600", "")
OUT = os.path.join(REPO, "docs", "figures")


def load_cells():
    """Train per-family heads on cal; return per-(A,B) test (task, label, decision-score)."""
    fams = {}
    for f in sorted(glob.glob(os.path.join(REPO, "results", T2, "*.npz"))):
        name = SHORT(os.path.basename(f)[:-4])
        d = np.load(f, allow_pickle=True)
        split = np.array([split_of(str(t)) for t in d["task_id"]])
        fams[name] = {"emb": d["emb"], "label": d["label"].astype(int),
                      "task": np.array([str(t) for t in d["task_id"]]), "split": split}
    names = sorted(fams)
    cal_emb = np.vstack([fams[a]["emb"][fams[a]["split"] == "cal"] for a in names])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    heads = {}
    for a in names:
        d = fams[a]; d["p"] = pca.transform(scaler.transform(d["emb"]))
        cal = d["split"] == "cal"
        heads[a] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    cells = {}
    for A in names:
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            cells[(A, B)] = (d["task"][te], d["label"][te], heads[A].decision_function(d["p"][te]))
    return names, cells


def auroc(cell):
    _, y, s = cell
    return roc_auc_score(y, s)


def boot_contrast(cells, pairs_hi, pairs_lo, n_boot=1000, seed=0):
    """Task-clustered bootstrap of mean(AUROC over pairs_hi) - mean(AUROC over pairs_lo)."""
    rng = np.random.default_rng(seed)
    all_tasks = sorted({t for p in (pairs_hi + pairs_lo) for t in cells[p][0]})
    diffs, his, los = [], [], []
    for _ in range(n_boot):
        keep = set(rng.choice(all_tasks, size=len(all_tasks), replace=True).tolist())
        def mean_auroc(pairs):
            vals = []
            for p in pairs:
                t, y, s = cells[p]; m = np.array([x in keep for x in t])
                if len(np.unique(y[m])) < 2:
                    return None
                vals.append(roc_auc_score(y[m], s[m]))
            return float(np.mean(vals))
        hi, lo = mean_auroc(pairs_hi), mean_auroc(pairs_lo)
        if hi is None or lo is None:
            continue
        his.append(hi); los.append(lo); diffs.append(hi - lo)
    a = np.array(diffs)
    return (np.mean(his), np.mean(los), np.mean(a),
            np.percentile(a, 2.5), np.percentile(a, 97.5))


def main():
    cap_raw = json.load(open(os.path.join(REPO, "results", "capability.json")))
    cap = {("frontier_sonnet" if k == "sonnet" else k): v["main_rate"] for k, v in cap_raw.items()}
    names, cells = load_cells()
    order = sorted(names, key=lambda n: cap[n])  # weakest -> strongest
    print("Capability (attack main-task success), weakest -> strongest:")
    for n in order:
        print(f"  {SHORT(n):16} {cap[n]:.3f}")
    strongest = order[-1]
    print(f"\nstrongest (frontier) = {strongest}\n")

    M = {(A, B): auroc(cells[(A, B)]) for A in names for B in names}

    # --- Headline A: deploy against the frontier (calibrate weak, test strong) ---
    open_to_front = [(A, strongest) for A in names if A != strongest]
    diag_front = [(strongest, strongest)]
    hi, lo, d, lo_ci, hi_ci = boot_contrast(cells, diag_front, open_to_front)
    print("=== Headline A: monitoring the frontier ===")
    print(f"  in-lineage (Sonnet head on Sonnet)         AUROC {hi:.3f}")
    print(f"  off-lineage (open-family heads on Sonnet)  AUROC {lo:.3f}")
    print(f"  GAP (in - off) = {d:+.3f}  95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]")
    for A in names:
        if A != strongest:
            print(f"    {SHORT(A):14}-calibrated -> Sonnet: AUROC {M[(A, strongest)]:.3f}")

    # --- Headline B: weak->strong vs strong->weak off-diagonal asymmetry ---
    w2s = [(A, B) for A in names for B in names if A != B and cap[A] < cap[B]]
    s2w = [(A, B) for A in names for B in names if A != B and cap[A] > cap[B]]
    hi2, lo2, d2, lo2c, hi2c = boot_contrast(cells, s2w, w2s)
    print("\n=== Headline B: transfer direction asymmetry ===")
    print(f"  strong->weak mean AUROC {hi2:.3f}   weak->strong mean AUROC {lo2:.3f}")
    print(f"  asymmetry (s2w - w2s) = {d2:+.3f}  95% CI [{lo2c:+.3f}, {hi2c:+.3f}]")

    # --- Trend: off-diagonal AUROC vs |capability gap| ---
    gaps, aurocs, dirs = [], [], []
    for A in names:
        for B in names:
            if A == B:
                continue
            gaps.append(cap[B] - cap[A]); aurocs.append(M[(A, B)])
            dirs.append("w2s" if cap[A] < cap[B] else "s2w")
    gaps, aurocs = np.array(gaps), np.array(aurocs)
    r = np.corrcoef(np.abs(gaps), aurocs)[0, 1]
    print(f"\nCorr(|capability gap|, off-diagonal AUROC) = {r:+.3f}")

    # --- Deployment catch-rate (safety JSON, budget-enforced q=1%) ---
    catch_in = catch_off = None
    sj = os.path.join(REPO, "results", "safety_translation_track2_n600_5fam.json")
    if os.path.exists(sj):
        be = json.load(open(sj))["variants"]["budget-enforced"]["0.01"]["TPR"]
        front = "frontier_sonnet_n600"
        catch_in = 100 * be[f"{front}->{front}"]
        catch_off = 100 * np.mean([be[f"{A}_n600->{front}"] for A in ["ow_gemma", "ow_llama", "ow_mistral", "ow_qwen"]])
        print("\n=== Deployment catch-rate against the frontier (q=1% budget) ===")
        print(f"  in-lineage {catch_in:.1f}%   weak-calibrated mean {catch_off:.1f}%")

    # --- Figure 4: deployment contrast (the money) + mechanism panel ---
    auroc_in = M[(strongest, strongest)]
    auroc_off = np.mean([M[(A, strongest)] for A in names if A != strongest])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    # panel 1: monitoring the frontier (AUROC + catch%)
    ax = axes[0]
    labels = ["in-lineage\n(Sonnet-calibrated)", "weak-calibrated\n(open families)"]
    ax.bar([0, 1], [auroc_in, auroc_off], color=["#2166ac", "#b2182b"], width=0.6)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.text(1.45, 0.505, "chance", color="grey", fontsize=7, va="bottom", ha="right")
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("AUROC on frontier attacks")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate([auroc_in, auroc_off]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    if catch_in is not None:
        ax.text(0.5, 0.43, f"catch @1%: {catch_in:.0f}% vs {catch_off:.0f}%", ha="center",
                fontsize=8, style="italic", transform=ax.get_xaxis_transform())
    ax.set_title("Monitoring a frontier attacker", fontsize=10)
    # panel 2: mechanism -- transfer is not a capability-distance effect
    ax = axes[1]
    g = np.array(gaps)
    ax.axhline(0.5, color="grey", lw=1, ls=":"); ax.axvline(0, color="grey", lw=1, ls="--")
    for dlabel, color, mark in [("w2s", "#b2182b", "o"), ("s2w", "#2166ac", "s")]:
        m = np.array([x == dlabel for x in dirs])
        ax.scatter(g[m], aurocs[m], c=color, marker=mark, s=50,
                   label=("calibrate weak, test strong" if dlabel == "w2s" else "calibrate strong, test weak"))
    ax.set_xlabel("capability gap (eval minus calibration main-success)")
    ax.set_ylabel("cross-family AUROC")
    ax.set_title("Transfer is not a capability-distance effect", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig4_capability_gap.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}/fig4_capability_gap.*")


if __name__ == "__main__":
    main()
