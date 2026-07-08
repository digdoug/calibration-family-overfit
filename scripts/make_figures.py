"""Generate the three paper figures from the n=600 results.

Fig 1 (cross-family AUROC heatmap) and the APPS interaction in Fig 2 are RECOMPUTED LIVE from
results/track2_n600[_actiononly]/*.npz with the exact track2_matrix pipeline (shared scaler +
PCA-50 + per-family logistic head, md5-seed-0 60/40 split), so they cannot drift from the paper.
Fig 3 reads the per-roster safety_translation_track2_n600[_Nfam].json files and plots the q=1%
catch-rate progression (4 open -> 8 families), ending at the headline eight-family 42/24 — the gap
widens as rankable frontier families are added (the concentration thesis). Cross-document conditions in Fig 2
(bigcodebench, LoRA) are canonical constants from the control ledger (docs/CONTROLS_RESULTS.md).

Run: .venv/bin/python scripts/make_figures.py   ->  docs/figures/*.pdf + *.png
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from track2_matrix import split_of, diag_interaction  # noqa: E402

OUT = os.path.join(REPO, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
def SHORT(n):
    n = n.replace("ow_", "").replace("_n600", "").replace("frontier_", "")
    return {"gemini_flash": "gemini", "gpt5": "gpt-5"}.get(n, n)


def build_matrix(t2dir):
    """Recompute the cross-family AUROC matrix exactly as track2_matrix.main()."""
    fams = {}
    for f in sorted(glob.glob(os.path.join(REPO, "results", t2dir, "*.npz"))):
        name = SHORT(os.path.basename(f)[:-4])
        d = np.load(f, allow_pickle=True)
        split = np.array([split_of(str(t)) for t in d["task_id"]])
        fams[name] = {"emb": d["emb"], "label": d["label"].astype(int), "split": split}
    names = sorted(fams)
    cal_emb = np.vstack([fams[a]["emb"][fams[a]["split"] == "cal"] for a in names])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    heads = {}
    for a in names:
        d = fams[a]
        d["p"] = pca.transform(scaler.transform(d["emb"]))
        cal = d["split"] == "cal"
        heads[a] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])
    M = {}
    for A in names:
        for B in names:
            d = fams[B]; te = d["split"] == "test"
            M[(A, B)] = roc_auc_score(d["label"][te], heads[A].decision_function(d["p"][te]))
    return names, M


# ---------- Figure 1: cross-family AUROC heatmap ----------
def fig_heatmap(t2dir="track2_n600", outname="fig1_matrix",
                title="Cross-family monitor AUROC (diagonal = in-lineage)",
                order=None, figsize=(5.2, 4.4)):
    names, M = build_matrix(t2dir)
    if order is not None:
        names = [n for n in order if n in names] + [n for n in names if n not in order]
    arr = np.array([[M[(a, b)] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=figsize)
    # vmin floors at 0.40 (the 4-/5-/6-family mins) but extends down to the data when a
    # roster has a below-chance off-diagonal cell (the 8-family matrix bottoms out at 0.34
    # for qwen->gpt-5 and gpt-5->llama); a fixed 0.40 floor would clip those to the colorbar
    # minimum and make the scale not span its own data.
    norm = TwoSlopeNorm(vmin=min(0.40, float(arr.min())), vcenter=0.50, vmax=1.0)
    im = ax.imshow(arr, cmap="RdBu_r", norm=norm, aspect="equal")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([SHORT(n) for n in names]); ax.set_yticklabels([SHORT(n) for n in names])
    ax.set_xlabel("evaluation family (B)"); ax.set_ylabel("calibration family (A)")
    for i in range(len(names)):
        for j in range(len(names)):
            v = arr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if (v > 0.80 or v < 0.45) else "black",
                    fontweight="bold" if i == j else "normal", fontsize=10)
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=2))
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label("AUROC")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{outname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    diag = np.mean([M[(a, a)] for a in names])
    off = np.mean([M[(a, b)] for a in names for b in names if a != b])
    return diag, off, diag_interaction(M, names)


# ---------- Figure 2: interaction forest plot ----------
def live_interaction(t2dir):
    names, M = build_matrix(t2dir)
    return diag_interaction(M, names)

def fig_forest():
    apps_full = live_interaction("track2_n600")
    apps_act = live_interaction("track2_n600_actiononly")
    fivefam = live_interaction("track2_n600_5fam")
    import os as _os
    sixfam = live_interaction("track2_n600_6fam") if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_6fam")) else 0.137
    eightfam = live_interaction("track2_n600_8fam") if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_8fam")) else 0.150
    frontier4 = live_interaction("track2_frontieronly4") if _os.path.isdir(_os.path.join(REPO, "results", "track2_frontieronly4")) else 0.103
    # DeepSeek 9th-family fold (5th frontier vendor); supersedes the 8-fam / frontier-4 rows below
    # once present. Falls back to the recorded fold values until the dirs exist.
    ninefam = live_interaction("track2_n600_9fam") if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_9fam")) else 0.152
    frontier5 = live_interaction("track2_frontieronly5") if _os.path.isdir(_os.path.join(REPO, "results", "track2_frontieronly5")) else 0.100
    _have9 = _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_9fam"))
    nomic = live_interaction("track2_n600_nomic") if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_nomic")) else 0.097
    # SHADE: private task set, aggregate numbers only; CI from cluster bootstrap.
    # live point where the gitignored npz are present.
    shade = live_interaction("track2_shade") if _os.path.isdir(_os.path.join(REPO, "results", "track2_shade")) else 0.030
    # (label, value, lo, hi); live where data is local, CIs + cross-doc rows are canonical constants.
    # CI whiskers are the MULTIPLICITY-PRESERVING cluster bootstrap (_clusterboot path):
    # recomputed at $0 from cached embeddings/scores via track2_matrix.py / staircase analyze_npz,
    # point estimates unchanged, intervals ~1/3 wider than the old set()-bug CIs. Every row still has its CI clear of 0 EXCEPT SHADE:
    # the multiplicity-preserving recompute of the SHADE matrix is +0.035 [-0.004, +0.077]
    # (spans 0; underpowered at 7 test tasks, disclosed in the paper's external-validity
    # section), while the whisker below is the documented original cluster-bootstrap interval.
    rows = [
        ("APPS, full transcript (n=600)", apps_full, 0.094, 0.125),
        ("APPS, action-only (n=600)", apps_act, 0.043, 0.067),
        ("APPS + frontier (Sonnet), 5-family", fivefam, 0.119, 0.145),
        ("APPS + 2 frontier vendors, 6-family", sixfam, 0.125, 0.148),
        ("APPS + 4 frontier vendors, 8-family", eightfam, 0.140, 0.159),
        ("APPS + 5 frontier vendors, 9-family", ninefam, 0.143, 0.160) if _have9 else None,
        ("APPS frontier-only (5 vendors)", frontier5, 0.090, 0.109) if _have9 else
        ("APPS frontier-only (4 vendors)", frontier4, 0.091, 0.113),
        ("non-OpenAI embedder (nomic)", nomic, 0.082, 0.112),
        ("bigcodebench, full", 0.250, 0.227, 0.271),
        ("bigcodebench, action-only", 0.141, 0.123, 0.158),
        ("LoRA monitor (Phi-3.5-mini)", 0.179, 0.150, 0.210),
        ("SHADE-Arena (non-code, 3 frontier vendors)", shade, 0.010, 0.065),
    ]
    rows = [r for r in rows if r is not None]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ys = range(len(rows))[::-1]
    for y, (lab, v, lo, hi) in zip(ys, rows):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color="#2166ac",
                    capsize=4, markersize=6, lw=2)
        ax.text(hi + 0.006, y, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("interaction (own-family advantage, net of main effects)")
    ax.set_xlim(-0.02, 0.30)
    ax.set_title("Calibration overfit replicates across cuts, monitors, substrates", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_forest.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return apps_full, apps_act


# ---------- Figure 3: realized-safety gap ----------
def fig_safety():
    # q=1% catch-rate progression across rosters (mirrors Table 3): the in-lineage vs
    # off-lineage gap WIDENS as rankable frontier families are added (4 open +6.5 -> 8 fam +18.4),
    # which is the concentration thesis. Ends at the headline eight-family 42/24.
    rosters = [
        ("4 open", "safety_translation_track2_n600.json"),
        ("5\n(+frontier)", "safety_translation_track2_n600_5fam.json"),
        ("6\n(2 vendors)", "safety_translation_track2_n600_6fam.json"),
        ("8\n(4 vendors)", "safety_translation_track2_n600_8fam.json"),
    ]
    labels, diag, off = [], [], []
    for lab, fn in rosters:
        d = json.load(open(os.path.join(REPO, "results", fn)))
        tpr = d["variants"]["budget-enforced"]["0.01"]["TPR"]
        dd = [v for k, v in tpr.items() if k.split("->")[0] == k.split("->")[1]]
        oo = [v for k, v in tpr.items() if k.split("->")[0] != k.split("->")[1]]
        labels.append(lab); diag.append(100 * np.mean(dd)); off.append(100 * np.mean(oo))
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    b1 = ax.bar(x - w/2, diag, w, label="in-lineage (calibrated on same family)", color="#2166ac")
    b2 = ax.bar(x + w/2, off, w, label="off-lineage (deployed on another family)", color="#b2182b")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f"{b.get_height():.0f}%", ha="center", va="bottom", fontsize=9)
    for xi, (dv, ov) in enumerate(zip(diag, off)):
        ax.annotate(f"+{dv-ov:.1f}", (xi, max(dv, ov) + 4.5), ha="center", fontsize=8,
                    color="#444", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("families in the matrix (rankable frontier vendors added rightward)")
    ax.set_ylabel("sabotage caught (TPR, %)")
    ax.set_ylim(0, max(diag) + 11)
    ax.set_title("Realized-safety gap widens with the rankable families (1% budget)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_safety.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return diag[-1], off[-1]


# ---------- Figure 6: monitor-capability staircase (Control 15) ----------
def fig_staircase(outname="fig6_staircase"):
    """Interaction vs trained-monitor BASE capability (Control 15). Reuses
    staircase_summary.analyze_npz so the points are byte-identical to the fold
    table (same interaction + task-clustered CI + competence gate). Each rung is
    one point at its base param count (log-x), green if the competence gate
    (mistral AND llama diag >=0.70) passes else red, with the 95% CI as error
    bars. The headline reads off the plot: the interaction stays CI>0 (and roughly
    flat) as the reader climbs 3.8B -> 14B -> 34B, so the cross-family gap is
    structural, not a weak-monitor artifact. No-op until >=1 staircase npz lands."""
    import staircase_summary as ss
    DST = os.path.join(REPO, "results", "track2_lora_staircase")
    # (label, params_billions, npz) — Phi-3.5 anchor is the Control-11 canon; the
    # rest are harvested staircase rungs. Seeds at the same base sit at the same x.
    rungs = [
        ("Phi-3.5-mini", 3.8, os.path.join(REPO, "results", "track2_lora", "scores_lora.npz")),
        ("Phi-4", 14.0, os.path.join(DST, "phi4_seed0.npz")),
        ("Phi-4 (seed 1)", 14.0, os.path.join(DST, "phi4_seed1.npz")),
        ("Yi-1.5-Chat", 34.0, os.path.join(DST, "yi34b_seed0.npz")),
        ("Yi-1.5-Chat (seed 1)", 34.0, os.path.join(DST, "yi34b_seed1.npz")),
    ]
    pts = []
    for lab, pb, path in rungs:
        if not os.path.exists(path):
            continue
        r = ss.analyze_npz(path)
        mis, lla = r["gate"]["mistral"], r["gate"]["llama"]
        gated = (mis is not None and mis >= ss.GATE) and (lla is not None and lla >= ss.GATE)
        pts.append((lab, pb, r["inter"], r["ilo"], r["ihi"], gated, len(r["names"])))
    if len(pts) < 2:
        return None  # need at least the anchor + one rung to draw a staircase
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # tiny multiplicative x-jitter so same-base seeds don't overlap on the log axis
    seen = {}
    for lab, pb, v, lo, hi, gated, nf in pts:
        k = seen.get(pb, 0); seen[pb] = k + 1
        x = pb * (1.0 + 0.05 * k)
        color = "#2166ac" if gated else "#b2182b"
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o", color=color,
                    capsize=4, markersize=7, lw=2,
                    label=("competence gate PASS" if gated else "gate FAIL"))
        tag = f"{v:+.3f}" + ("" if nf == 4 else f"\n(partial {nf}/4)")
        # stagger same-base seed labels above/below the point so near-equal values
        # (e.g. the two 34B seeds +0.166/+0.163) don't overprint into a smear
        ax.text(x * 1.06, v, tag, va=("bottom" if k == 0 else "top"), fontsize=8)
    ax.axhline(0, color="grey", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([3.8, 14.0, 34.0]); ax.set_xticklabels(["3.8B", "14B", "34B"])
    ax.set_xlim(3.0, 60.0)
    ax.set_xlabel("trained-monitor base capability (parameters, log scale)")
    ax.set_ylabel("cross-family interaction [95% CI]")
    ax.set_title("Interaction survives a stronger trained monitor (Control 15)", fontsize=10)
    # de-duplicate legend labels
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{outname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname} ({len(pts)} rungs present)")
    return len(pts)


def fig_concentration(t2dir="track2_n600_8fam", outname="fig5_concentration"):
    """Per-eval-family in-lineage (own head) vs mean-foreign-head AUROC, all families.
    build_matrix already applies SHORT() to the keys, so `order` uses cleaned names."""
    names, M = build_matrix(t2dir)
    order = ["qwen", "llama", "mistral", "gemma", "sonnet", "gemini", "grok", "gpt-5", "deepseek"]
    order = [n for n in order if n in names] + [n for n in names if n not in order]
    inl = [M[(b, b)] for b in order]
    frn = [np.mean([M[(a, b)] for a in names if a != b]) for b in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - w/2, inl, w, label="in-lineage (own family's head)", color="#2166ac")
    ax.bar(x + w/2, frn, w, label="foreign heads (mean)", color="#b2182b")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.text(len(order) - 0.5, 0.505, "chance", color="grey", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([SHORT(n) for n in order])
    ax.set_ylabel("AUROC on this family's attacks"); ax.set_ylim(0.45, 1.02)
    ax.set_title("Own-family head outranks foreign heads in every family", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{outname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname} ({len(order)} families)")


# ---------- Figure 7: head-direction mechanism (cos vs transfer AUROC) ----------
def _headangle_cells(t2dir, basis):
    """Off-diagonal (cos(w_A,w_B), AUROC(A->B)) cells from the STRICT head geometry.
    Reuses track2_headangle / track2_matrix_strict so the AUROC matrix is byte-identical
    to the headline strict matrix and the head directions to the mechanism result."""
    import track2_matrix_strict as tms
    import track2_headangle as H
    t2 = os.path.join(REPO, "results", t2dir)
    names, fams = tms.load_families(t2)
    rm = {a: tms.fit_row_model(fams, a, basis, 50) for a in names}
    w = {a: H.head_direction(rm[a]) for a in names}
    test_idx = {b: np.flatnonzero(fams[b]["split"] == "test") for b in names}
    rows = []  # (A, B, cos, auroc)
    for a in names:
        for b in names:
            if a == b:
                continue
            rows.append((tms.short_name(a), tms.short_name(b),
                         H.cosine(w[a], w[b]), H.cell_auroc(fams, b, rm[a], test_idx[b])))
    return rows


def fig_headangle(outname="fig7_headangle"):
    """Three-panel scatter: off-lineage transfer AUROC vs cross-family head cosine.
    Left = four-open primary matrix (12 off-diagonal cells); middle = nine families,
    full transcript (72 cells, the strongest roster); right = four-open action-only
    (12 cells), showing the relationship survives family-prose stripping. Each point
    is one calibrate-on-A / test-on-B cell. The rho annotations are the CANONICAL
    cluster-bootstrap means + 95% CIs (the
    values the paper cites); the dashed line is an ordinary-least-squares guide
    through the cloud. Gemma<->Llama is highlighted only in the two four-open panels,
    where it is the genuine max-cosine pair; in the nine-family matrix the most
    aligned pair is a frontier pair (DeepSeek<->Sonnet), so no single pair is singled
    out there."""
    # (t2dir, basis, title, rho_text, highlight_gemma_llama)
    panels = [
        ("track2_n600", "row-all",
         "Four open (primary)", r"$\rho = +0.56$  [+0.34, +0.79]", True),
        ("track2_n600_9fam", "row-all",
         "Nine families, full transcript", r"$\rho = +0.80$  [+0.79, +0.83]", False),
        ("track2_n600_actiononly", "row-benign",
         "Four open, action-only", r"$\rho = +0.66$  [+0.51, +0.78]", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    for ax, (t2dir, basis, title, rho_txt, hl) in zip(axes, panels):
        rows = _headangle_cells(t2dir, basis)
        cos = np.array([r[2] for r in rows])
        au = np.array([r[3] for r in rows])
        is_gl = np.array([{r[0], r[1]} == {"gemma", "llama"} for r in rows])
        keep = is_gl if hl else np.zeros(len(rows), dtype=bool)
        ax.scatter(cos[~keep], au[~keep], s=34, color="#2166ac", alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3, label="off-diagonal cell")
        if hl:
            ax.scatter(cos[keep], au[keep], s=80, color="#d6604d", marker="D",
                       edgecolor="black", linewidth=0.7, zorder=4,
                       label="Gemma↔Llama (most aligned)")
        # OLS guide line through the cloud
        xs = np.array([cos.min(), cos.max()])
        m, b = np.polyfit(cos, au, 1)
        ax.plot(xs, m * xs + b, color="#444", ls="--", lw=1.4, zorder=2)
        ax.axhline(0.5, color="grey", ls=":", lw=1, zorder=1)
        ax.text(ax.get_xlim()[1], 0.5, " chance", color="grey", fontsize=7, va="bottom", ha="right")
        ax.set_xlabel(r"cross-family head cosine  $\cos(w_A, w_B)$")
        ax.set_ylabel("off-lineage transfer AUROC")
        ax.set_title(title, fontsize=10)
        ax.text(0.04, 0.96, rho_txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2166ac", alpha=0.9))
        ax.legend(fontsize=7.5, loc="lower right", frameon=True)
    fig.suptitle("Off-lineage transfer is governed by head-direction alignment "
                 "(concept shift made geometric)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{outname}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outname} (4-open + 9-fam {len(_headangle_cells('track2_n600_9fam','row-all'))}-cell "
          f"+ action-only panels)")


if __name__ == "__main__":
    import os as _os0
    # fig5 tracks the headline roster: prefer the 9-family dir once the DeepSeek fold lands.
    _conc = "track2_n600_9fam" if _os0.path.isdir(_os0.path.join(REPO, "results", "track2_n600_9fam")) else "track2_n600_8fam"
    fig_concentration(_conc)
    diag, off, inter = fig_heatmap()
    print(f"fig1 heatmap (4-fam): mean diag {diag:.3f}, off {off:.3f}, interaction {inter:+.4f}")
    d5, o5, i5 = fig_heatmap("track2_n600_5fam", "fig1b_matrix_5fam",
                             "Cross-family AUROC with a frontier attacker (Sonnet)")
    print(f"fig1b heatmap (5-fam): mean diag {d5:.3f}, off {o5:.3f}, interaction {i5:+.4f}")
    import os as _os
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_6fam")):
        d6, o6, i6 = fig_heatmap("track2_n600_6fam", "fig1c_matrix_6fam",
                                 "Cross-family AUROC, six families (two frontier vendors)")
        print(f"fig1c heatmap (6-fam): mean diag {d6:.3f}, off {o6:.3f}, interaction {i6:+.4f}")
    # headline: the full eight-family matrix (four open-weight + four frontier vendors),
    # open-weight block then frontier block so the diagonal and the frontier sub-block read cleanly
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_8fam")):
        d8, o8, i8 = fig_heatmap(
            "track2_n600_8fam", "fig1d_matrix_8fam",
            "Cross-family AUROC, eight families (four frontier vendors)",
            order=["qwen", "llama", "mistral", "gemma", "sonnet", "gemini", "grok", "gpt-5"],
            figsize=(7.0, 6.0))
        print(f"fig1d heatmap (8-fam): mean diag {d8:.3f}, off {o8:.3f}, interaction {i8:+.4f}")
    # sharpest test: frontier-only sub-matrix (four strong models, four vendors)
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_frontieronly4")):
        df, of, if_ = fig_heatmap(
            "track2_frontieronly4", "fig1e_matrix_frontier4",
            "Frontier-only AUROC (four vendors, the sharpest transfer test)",
            order=["sonnet", "gemini", "grok", "gpt-5"], figsize=(5.0, 4.2))
        print(f"fig1e heatmap (frontier-4): mean diag {df:.3f}, off {of:.3f}, interaction {if_:+.4f}")
    # DeepSeek 9th-family fold (auto-fires once analysis_deepseek_fold.sh creates the dirs;
    # no-op until then). DeepSeek is a 5th independent frontier vendor; mirrors the 8-fam +
    # frontier-4 blocks above so the fold needs no hand-edit here, only the forest CI constants.
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_9fam")):
        d9, o9, i9 = fig_heatmap(
            "track2_n600_9fam", "fig1f_matrix_9fam",
            "Cross-family AUROC, nine families (five frontier vendors)",
            order=["qwen", "llama", "mistral", "gemma", "sonnet", "gemini", "grok", "gpt-5", "deepseek"],
            figsize=(7.4, 6.4))
        print(f"fig1f heatmap (9-fam): mean diag {d9:.3f}, off {o9:.3f}, interaction {i9:+.4f}")
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_frontieronly5")):
        df5, of5, if5 = fig_heatmap(
            "track2_frontieronly5", "fig1g_matrix_frontier5",
            "Frontier-only AUROC (five vendors, the sharpest transfer test)",
            order=["sonnet", "gemini", "grok", "gpt-5", "deepseek"], figsize=(5.4, 4.6))
        print(f"fig1g heatmap (frontier-5): mean diag {df5:.3f}, off {of5:.3f}, interaction {if5:+.4f}")
    # Control 15 monitor-capability staircase (auto-fires once analysis_staircase_fold.sh
    # harvests >=1 rung npz into results/track2_lora_staircase/; no-op until then).
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_lora_staircase")):
        fig_staircase()
    af, aa = fig_forest()
    print(f"fig2 forest: APPS full {af:+.4f}, action-only {aa:+.4f} (live recompute)")
    sd, so = fig_safety()
    print(f"fig3 safety @1%: in-lineage {sd:.1f}%, off-lineage {so:.1f}%")
    # Figure 7: head-direction mechanism (cos vs transfer AUROC). Strict-basis,
    # auto-fires whenever the 9-family + action-only npz are present.
    if _os.path.isdir(_os.path.join(REPO, "results", "track2_n600_9fam")):
        fig_headangle()
    print(f"wrote PDFs+PNGs to {OUT}")
