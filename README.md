# Calibration-Family Overfit: Why Trusted Sabotage Monitors Don't Transfer Across Lineages

Analysis code and aggregate results for an empirical AI-control study. The paper is
`paper/paper.pdf` (arXiv id: TBD; the link will be added when the preprint is announced);
all aggregate figures are in `docs/figures/`.

## The claim

Trusted monitoring is a central defense in AI control: a cheaper trusted model scores an untrusted
model's actions for sabotage, and the most suspicious are audited or deferred. Reported accuracies
are usually measured against one or two untrusted models and assumed to hold for others. We make the
untrusted policy *family* the controlled axis. For each calibration family A we fit a
family-conditioned monitor, apply it to a different family B, and decompose the cross-family AUROC
into a per-attacker effect (how obvious each family's sabotage is), a per-monitor effect (how capable
each monitor is), and an **interaction** term.

The interaction is positive and significant: a monitor calibrated on one family ranks its own
family's sabotage far above an equally blatant attack from a different lineage, net of attacker
obviousness and monitor capability. This is a control-**evaluation** / measurement-validity finding
(blue-team), not an attack paper. The prescription is that control evaluations should report
cross-family transfer matrices, because a single-pairing accuracy overstates the safety a monitor
delivers against a policy lineage it was not calibrated on.

## Result at a glance

The paper PDF is the source of record for the current numbers. Headline figures, with their feature
basis labeled (the strict leak-free basis is primary; the transductive basis is the conservative
lower-variance read):

- **Interaction +0.172** (95% CI [+0.158, +0.185], strict leak-free; +0.110 [+0.094, +0.125]
  transductive) on four open-weight families at 600 transcripts per family, the preregistered
  primary roster. On expanded post-hoc rosters the strict interaction plateaus near +0.19:
  +0.197 on eight families (four frontier vendors), +0.194 on nine (adding DeepSeek, a fifth
  vendor).
- **Not a weak-open-model artifact:** restricted to the frontier lineages alone the interaction
  is about **+0.10** full / **+0.135** action-only (code-only, prose stripped), the behavioral cut
  if anything larger.
- **Significant, not just stable:** across the eight families the observed diagonal is the single most
  extreme of all 40,320 family matchings (exact permutation p = 0.000025).
- **Monitor-class robust:** a LoRA-finetuned monitor that reads the raw transcript replicates at
  **+0.18** (three seeds, two base models). **Substrate robust:** replicates on bigcodebench at
  **+0.233** strict (+0.250 transductive).
- **Deployment cost:** under defer-to-trusted at a 1% audit budget on the eight-family matrix, an
  off-lineage monitor catches **19%** of attack attempts where an in-lineage monitor catches
  **41%**, a **+21.7-point gap at the same audit cost** ([+19.7, +26.7], strict; 42% vs 24%
  transductive). Catch rates measure attack-attempt detection, not realized end-to-end sabotage.

## Two metric tracks (why this is not a math error)

A fixed prompt-only monitor produces the same score *ranking* regardless of which family you
calibrated on, so its **AUROC** matrix has no calibration-family dependence. The headline AUROC
transfer matrix therefore uses a **family-conditioned** monitor whose scores depend on the
calibration family (a frozen-embedding + logistic head trained on family A, and a LoRA monitor as the
second class). Two tracks are reported throughout:

- **Track 1, operational calibration transfer** (`scripts/w2_run_matrix.py`, `scripts/track2_safety.py`):
  threshold set at the 99th percentile of family A's benign suspicion, applied to family B; reported as
  TPR@1%FPR and realized safety. Any monitor, including the fixed prompt monitor.
- **Track 2, ranking transfer** (`scripts/track2_embed.py` + `scripts/track2_matrix.py`): the AUROC
  interaction matrix. The scoring function depends on the calibration family by construction (asserted
  in code).

## Reproducing the analysis

Generation and analysis are deliberately decoupled (**generate once, score many**): trajectories are
generated once per policy family, then scored offline by many monitors. Re-running the matrix or any
control does **not** require regenerating trajectories.

### Environments

Two virtualenvs keep the headline-statistic dependency surface small:

```bash
# analysis core (numpy/scipy/sklearn/pandas): all of Track-2 and the controls
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt

# agent runtime (ControlArena + Inspect AI + Docker): only for generation
python3.11 -m venv .venv-controlarena
git clone https://github.com/UKGovernmentBEIS/control-arena vendor/control-arena
git -C vendor/control-arena checkout 3369a0915ac63353615c90318afce4eddb2c5efd  # the pinned SHA
.venv-controlarena/bin/pip install -e vendor/control-arena --no-deps
```

ControlArena is pinned in `configs/provenance.yaml` (commit SHA, pip version, applied patches).
Monitor scoring needs `OPENAI_API_KEY` (embeddings); generation needs the relevant policy-host key
(`DEEPINFRA_API_KEY`, `XAI_API_KEY`, etc.). Keys live in `.env` (gitignored).

### From transcripts to the headline number

```bash
# DIRS = the positional family result-dir names under results/ (e.g. W2_ow_qwen_n600 ...)

# 1. embed transcripts -> per-family score arrays (full and action-only).
#    Runs in .venv-controlarena because it reads the eval logs via control_arena.
.venv-controlarena/bin/python scripts/track2_embed.py --out track2_n600 $DIRS
.venv-controlarena/bin/python scripts/track2_embed.py --action-only --out track2_n600_actiononly $DIRS

# 2. calibrate-on-A / test-on-B matrix -> interaction + task-clustered bootstrap CI (analysis venv)
.venv/bin/python scripts/track2_matrix.py track2_n600
.venv/bin/python scripts/track2_matrix.py track2_n600_actiononly

# 3. figures (live-recompute from the npz)
.venv/bin/python scripts/make_figures.py
```

The single-command fold scripts (`scripts/analysis_*.sh`) chain embed → matrix → permutation test →
OOD/recal controls → safety for a given roster; `analysis_frontier_fold.sh` is the eight-family
pipeline, `analysis_deepseek_fold.sh` the nine-family extension.

### Generating trajectories (optional; the expensive part)

```bash
.venv-controlarena/bin/python scripts/w2_generate_family.py \
  --family ow_mistral --untrusted-model openai/mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --setting apps --modes honest attack --limit 600 --max-sandboxes 6
```

Long unattended runs use `scripts/run_family_pipeline.sh` (durable nohup, sample-count zombie
detection, Docker hard-clean, auto-caffeinate); its header comments document the launch procedure
and the failure-mode traps.

## Code map (result → script)

Every headline result and control has a runnable script; details and exact numbers are in
`docs/CONTROLS_RESULTS.md`.

| Result | Script |
|---|---|
| Track-1 operational matrix (fixed-prompt) | `scripts/w2_run_matrix.py` |
| Track-2 monitor + interaction matrix + bootstrap | `scripts/track2_embed.py`, `scripts/track2_matrix.py` |
| Ctrl 1 fixed-prompt FPR ablation / Ctrl 2 family-ID / Ctrl 3 LOFO pooling | `scripts/track2_controls.py` |
| Ctrl 4 difficulty-matching | `scripts/track2_difficulty.py` |
| Ctrl 5 / Ctrl 8 few-shot LLM monitor (competence ladder) | `scripts/track2_fewshot.py`, `scripts/track2_fewshot_v2.py` |
| Ctrl 6 head-overfit ablation / Ctrl 7 embedder-specificity | `scripts/track2_ablation.py`, `scripts/track2_embed.py --emb-model` |
| Ctrl 9 family-invariant scoring (mitigation) | `scripts/track2_invariant.py` |
| Ctrl 10 head-class robustness | `scripts/track2_headclass.py` |
| Ctrl 11 trained LoRA monitor (second monitor class) | `scripts/track2_lora_monitor.py` |
| Ctrl 12 bigcodebench second substrate | `docs/CONTROLS_RESULTS.md` (Control 12) |
| Ctrl 13 realized-safety translation | `scripts/track2_safety.py` |
| Ctrl 14 blatancy-stratified replication | `scripts/track2_blatancy.py` |
| Ctrl 16 drop-one-family jackknife | `scripts/track2_jackknife.py` |
| Ctrl 17 exact permutation test | `scripts/track2_permtest.py` |
| Ctrl 18 benign-only transfer predictor | `scripts/track2_predict.py` |
| Ctrl 19 inference-time OOD-deferral gate | `scripts/track2_ood_gate.py` |
| Ctrl 20 benign-only recalibration (concept vs covariate shift) | `scripts/track2_recalibrate.py` |
| Ctrl 21 recalibration learning curve | `scripts/track2_recal_curve.py` |
| Ctrl 22 cross-architecture cell concordance | `scripts/track2_concordance.py` |
| Ctrl 23 multi-family calibration curve | `scripts/track2_multifam_cal.py` |
| §4.6 capability-gap reanalysis | `scripts/capability_gap.py` |
| Figures | `scripts/make_figures.py` |

## Data handling and responsible release

This repository releases **analysis code and aggregate results only**. It does **not** contain raw
sabotage transcripts:

- `results/` and `trajectories/` (raw transcripts, monitor scores, score arrays) are gitignored and
  are not part of the release. Only aggregate figures (`docs/figures/`) and the compiled paper are
  committed.
- ControlArena and SHADE-Arena **canary strings** are honored. SHADE-Arena's gated task set is
  used under an aggregate-only condition: public artifacts carry summary statistics (matrices,
  CIs), never raw transcripts or task text.
- The work uses **stock honest/attack prompts** shipped with ControlArena. It contains no
  monitor-targeted prompt injection, no iterative attack optimization, and no evasion toolkit; the
  threat model is passive distribution shift, not adaptive attack.

To reproduce from scratch you regenerate trajectories yourself (the generation commands above) against
your own API keys; the saved score arrays the figures are built from are not redistributed.

## Provenance

`configs/provenance.yaml` is the single source of truth: ControlArena commit SHA and pip version,
Python and OS versions, the applied test-timeout patch, and the model/setting pins. `requirements.txt`
documents the dependency spec; `requirements.lock.txt` is the installed lock.

## Status

Headline at paper-n (600 transcripts per family); the full matrix spans nine families
across five frontier vendors. The paper PDF is the source of record for all numbers;
`docs/CONTROLS_RESULTS.md` is the full control ledger.
