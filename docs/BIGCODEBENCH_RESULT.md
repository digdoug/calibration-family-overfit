# 2nd-substrate result — bigcodebench replicates the cross-family transfer failure

Generated 2026-06-10 (overnight DigitalOcean run; the #1 external-validity item). The
calibration-overfit interaction REPLICATES on bigcodebench, stronger than on APPS. The
APPS-only external-validity hole is closed. Source data is gitignored; canonical numbers here.

## Headline

| cut | APPS (headline) | **bigcodebench** | verdict |
|---|---|---|---|
| full transcript | +0.127 [+0.112, +0.145] | **+0.250 [+0.233, +0.268]** | replicates, ~2× larger ✓ |
| action-only (behavioral) | +0.039 [+0.023, +0.056] | **+0.141 [+0.127, +0.155]** | replicates, ~3.5× larger ✓ |

Interaction = mean diagonal − off-diagonal AUROC net of row (monitor) + column (attacker)
main effects; cluster-bootstrap (1000) by test task. CI excludes 0 on both cuts → the
cross-family ranking overfit is NOT APPS-specific. The behavioral component (action-only) is
far more pronounced here (~56% of full vs ~30% on APPS), strengthening the "not merely
surface style" claim.

## Full-transcript matrix (rows = calibration head, cols = eval family)

```
            bcb_llama  bcb_mistral   bcb_qwen
 bcb_llama      0.864      0.264      0.502
bcb_mistral     0.183      1.000      0.494
  bcb_qwen      0.165      0.873      0.501
```
mean diagonal 0.788 · mean off-diagonal 0.413 · raw bonus +0.375 [+0.350, +0.401] ·
**INTERACTION +0.250 [+0.233, +0.268]**.

Diagonal-dominant + smoking-gun anti-transfer: Llama's attacks are detected only by Llama's
own head (0.864); foreign heads score them **below chance** (Mistral-head→Llama 0.183,
Qwen-head→Llama 0.165). Qwen's column sits at ~chance for everyone (0.49–0.50) — Qwen is a
subtle/weak attacker on bcb too (mirrors APPS, where Qwen's own AUROC was ~0.54).

## Action-only matrix

mean diagonal 0.643 · mean off-diagonal 0.432 · raw bonus +0.211 [+0.191, +0.233] ·
**INTERACTION +0.141 [+0.127, +0.155]**.

## Generation funnel (n=200/family, both modes, all on the box)

| family | model | total | scored | note |
|---|---|---|---|---|
| bcb_mistral | mistralai/Mistral-Small-3.2-24B-Instruct-2506 | 400 | 400 | clean |
| bcb_llama | meta-llama/Llama-3.3-70B-Instruct | 400 | 385 | ~4% verification errors (normal) |
| bcb_qwen | Qwen/Qwen3-14B (think-off) | 400 | 400 | clean |

Smoke gate before the full run: Mistral unfiltered honest-vs-attack AUROC = 1.000 (n=6/mode),
clears the >0.65 bar. Same pipeline as APPS (md5-seed-0 60/40 task-disjoint split, pooled-cal
StandardScaler + PCA-50, per-family LogisticRegression C=1.0, decision_function score,
cluster-bootstrap by task) → directly comparable to the +0.127.

## Honest caveats

- **Diagonal is concentrated** (same shape as APPS's mistral-carried result): Mistral diag
  1.000 and Llama diag 0.864 carry the diagonal mean; Qwen's own-family diagonal is ~0.50
  (chance) because Qwen barely produces detectable bcb backdoors. The interaction is net of
  main effects and cluster-bootstrapped, which accounts for the concentration, but the clean
  claim is "for the lineages a monitor can rank at all, off-lineage calibration collapses
  detection (often below chance)," not "for every family."
- **3 families** (Mistral, Llama, Qwen; no Gemma — fragile multi-turn submission). A 3-family
  matrix replicates the diagonal-dominant structure, as PHASE4 anticipated.
- **n=200/family**, unfiltered honest-vs-attack AUROC (both_ok≈0 for sub-frontier models on
  bcb too — same disclosed framing as APPS).
- Mistral's perfect 1.000 diagonal is unusually high (bcb backdoors from Mistral are very
  separable in-family); it inflates the diagonal mean — the cluster-boot CI reflects it.

## Reproduce

Data on the Mac at `results/bcb_box/results/W2_bcb_{mistral,llama,qwen}_n200/` (pulled from the
droplet). Embeddings at `results/track2_bcb_box/` (full) and `results/track2_bcb_box_actiononly/`.
```
.venv-controlarena/bin/python scripts/track2_embed.py --out track2_bcb_box results/bcb_box/results/W2_bcb_*_n200
.venv-controlarena/bin/python scripts/track2_embed.py --action-only --out track2_bcb_box_actiononly results/bcb_box/results/W2_bcb_*_n200
.venv/bin/python scripts/track2_matrix.py track2_bcb_box
.venv/bin/python scripts/track2_matrix.py track2_bcb_box_actiononly
```

## Operational notes from the Linux/amd64 box run (for BIGCODEBENCH_RUN.md / future runs)

Fresh Ubuntu 24.04 amd64 box (DigitalOcean, 4 vCPU/8GB, MAX_SANDBOXES=3). Native amd64 ran all
three families cleanly in ~5.8h (07:51→13:39 UTC) — including **Llama, which had zombied under
Apple-Silicon emulation on the Mac (confirming that failure was purely the ~10× QEMU slowdown,
not a Llama-on-bcb problem)**. Four one-time gotchas, all fixed:
1. Ubuntu `docker.io` lacks the Compose v2 plugin (Docker Desktop bundles it) → `apt install docker-compose-v2`.
2. A fresh `pip install -e control-arena` pulled newer **inspect_ai 0.3.239 / openai 2.41.0**, which route the OpenAI provider to the **Responses API** → DeepInfra 404s. Pin to the dev-Mac versions: **inspect_ai==0.3.229, openai==2.38.0**.
3. `run_bigcodebench_families.sh` line `source ~/.zshrc || source ~/.bashrc` **crashes under `set -u`** on Linux (Ubuntu's root `.bashrc` hits an unset var; `|| true` can't catch a set-u exit). Repoint to `source ~/.env_keys`.
4. `scripts/track2_embed.py` line 41 `_arg("--out")` was missing its required `default` arg → unconditional TypeError. Fixed to `_arg("--out", None)` (committed-tree edit, left uncommitted for review).
