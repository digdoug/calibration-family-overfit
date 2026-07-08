"""Optional deepening: difficulty-matching control re-run on the 5-family n=600 data (Control 4 was
n=200/4-family). Neutralizes the attacker-capability/obviousness confound at the new scope: per eval
family, subsample TEST-split attacks to an identical main-task-success composition (per-stratum min
across families), keep honest negatives in full, re-score the SAME heads, recompute the interaction.

Self-contained (does not touch the canonical track2_difficulty.py): reads main-success per task from
results/capability.json and embeddings from results/track2_n600_5fam. Run with .venv.
"""
import os, glob, json, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from track2_matrix import split_of  # noqa: E402

SHORT = lambda n: n.replace("ow_", "").replace("_n600", "")
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0


def diag_interaction(cells, names):
    arr = np.array([[cells[(a, b)] for b in names] for a in names])
    grand = arr.mean(); row = arr.mean(axis=1); col = arr.mean(axis=0)
    return float(np.mean([arr[i, i] - row[i] - col[i] + grand for i in range(len(names))]))


# capability.json keys -> SHORT family names (matching load(), which strips _n600)
CAPKEY = {"mistral": "mistral", "llama": "llama", "qwen": "qwen",
          "gemma": "gemma", "sonnet": "frontier_sonnet"}


def load():
    cap = json.load(open(os.path.join(REPO, "results", "capability.json")))
    succ = {}
    for k, v in cap.items():
        succ[CAPKEY[k]] = {str(t): bool(int(m)) for t, m in v["main_per_task"].items()}
    fams = {}
    for f in sorted(glob.glob(os.path.join(REPO, "results", "track2_n600_5fam", "*.npz"))):
        name = SHORT(os.path.basename(f)[:-4])
        d = np.load(f, allow_pickle=True)
        fams[name] = {"emb": d["emb"], "label": d["label"].astype(int),
                      "task": np.array([str(t) for t in d["task_id"]]),
                      "split": np.array([split_of(str(t)) for t in d["task_id"]])}
    return fams, sorted(fams), succ


def build_matched(fams, names, succ, seed):
    avail = {}
    for B in names:
        d = fams[B]
        te = (d["split"] == "test") & (d["label"] == 1)
        tasks = d["task"][te]; s = succ[B]
        avail[B] = ([t for t in tasks if s.get(t, False)], [t for t in tasks if not s.get(t, False)])
    tgt_T = min(len(T) for T, F in avail.values()); tgt_F = min(len(F) for T, F in avail.values())
    print(f"matched composition: main-pass {tgt_T} + main-fail {tgt_F} = {tgt_T+tgt_F}/family "
          f"(rate {tgt_T/(tgt_T+tgt_F)*100:.1f}%)")
    rng = np.random.default_rng(seed)
    for B in names:
        T, F = avail[B]
        sel = {T[i] for i in rng.choice(len(T), tgt_T, replace=False)} | \
              {F[i] for i in rng.choice(len(F), tgt_F, replace=False)}
        keep = np.ones(len(fams[B]["task"]), dtype=bool)
        for i, (t, lab, sp) in enumerate(zip(fams[B]["task"], fams[B]["label"], fams[B]["split"])):
            if sp == "test" and lab == 1 and t not in sel:
                keep[i] = False
        fams[B]["keep"] = keep


def main():
    fams, names, succ = load()
    build_matched(fams, names, succ, SEED)
    cal_emb = np.vstack([fams[a]["emb"][fams[a]["split"] == "cal"] for a in names])
    scaler = StandardScaler().fit(cal_emb)
    pca = PCA(n_components=50, random_state=0).fit(scaler.transform(cal_emb))
    heads = {}
    for a in names:
        d = fams[a]; d["p"] = pca.transform(scaler.transform(d["emb"]))
        cal = d["split"] == "cal"
        heads[a] = LogisticRegression(C=1.0, max_iter=3000).fit(d["p"][cal], d["label"][cal])

    def cell_mask(B):
        d = fams[B]
        return (d["split"] == "test") & ((d["label"] == 0) | d["keep"])

    cells = {}
    for A in names:
        for B in names:
            d = fams[B]; m = cell_mask(B)
            cells[(A, B)] = roc_auc_score(d["label"][m], heads[A].decision_function(d["p"][m]))
    inter = diag_interaction(cells, names)
    diag = np.mean([cells[(x, x)] for x in names])
    off = np.mean([cells[(a, b)] for a in names for b in names if a != b])

    # cluster bootstrap over matched test tasks
    rng = np.random.default_rng(SEED)
    test_tasks = sorted({t for B in names for t, sp, k in
                         zip(fams[B]["task"], fams[B]["split"], fams[B]["keep"]) if sp == "test" and k})
    ints = []
    for _ in range(1000):
        keepset = set(rng.choice(test_tasks, len(test_tasks), replace=True).tolist())
        c, ok = {}, True
        for A in names:
            for B in names:
                d = fams[B]
                m = cell_mask(B) & np.array([t in keepset for t in d["task"]])
                y = d["label"][m]
                if len(np.unique(y)) < 2:
                    ok = False; break
                c[(A, B)] = roc_auc_score(y, heads[A].decision_function(d["p"][m]))
            if not ok:
                break
        if ok:
            ints.append(diag_interaction(c, names))
    ints = np.array(ints)
    print(f"\nmatched interaction = {inter:+.4f}  95% CI [{np.percentile(ints,2.5):+.4f}, {np.percentile(ints,97.5):+.4f}]")
    print(f"  mean diag {diag:.3f} / off {off:.3f}   (vs unmatched 5-family +0.1325)")
    print(f"  bootstrap mean {ints.mean():+.4f}, n={len(ints)}")


if __name__ == "__main__":
    main()
