# Realized-safety translation of the cross-family transfer failure

> CI CORRECTION (s169, 2026-06-15): the CIs in this doc were computed with the
> `set()` cluster bootstrap that drops draw multiplicity and runs ~1/3 too narrow.
> The multiplicity-corrected CIs (point estimates identical) are in
> `docs/SETFIX_OLDNEW_DELIVERABLE.md`: budget-enforced q=1% DROP +11.3 [+5.2, +14.6],
> interaction +8.5 [+3.9, +11.0] (both still clear of 0). Use those whisker widths.
> The strict non-transductive pipeline is `scripts/track2_safety_strict.py`.

Updated 2026-06-09 (moonshot session). Converts the AUROC interaction (+0.127, the
headline) into the number a deployed control protocol actually delivers: the fraction
of sabotage a defer-to-trusted monitor catches under a fixed audit budget. Reviewers
discount "your AUROC metric is off"; they cannot discount "your deployment safety
guarantee is off by this much at the same audit cost." Source: `scripts/track2_safety.py`
(copies the exact track2_matrix pipeline: md5-seed-0 60/40 task-disjoint split, pooled-cal
StandardScaler + PCA-50, per-family logistic head C=1.0, decision_function as score,
cluster-bootstrap by test task). Data is gitignored; canonical numbers recorded here.

## TL;DR (the money quote)

> Under a defer-to-trusted protocol that audits the top **1%** most-suspicious actions
> (the standard AI-control audit budget), a family-conditioned sabotage monitor catches
> **21%** of in-lineage sabotage but only **10%** of equally-successful off-lineage sabotage
> **at the same 1% audit cost** — an **11-point realized-safety gap** (95% CI [+6.5, +14.1],
> cluster-bootstrap by task), i.e. an off-lineage attacker is about **2.2× more likely to
> slip past auditing** than the in-lineage attacker the monitor's numbers were measured on.
> The gap is the main-effects-removed interaction **+8.5 pts [+4.8, +10.6]**, survives
> difficulty-matching (+11.2 pts), and is the deployment-facing image of the +0.127 AUROC
> interaction. Worse, a **static** threshold shipped from the calibration family doesn't even
> hold the budget off-lineage: realized audit spend on B's benign traffic ranges 0–100%
> (mean 44%) against a 1% target, so the operating point itself is family-specific.

## Why there are three protocol variants (and which one is the safety number)

"Calibrated to a q% audit budget" can break in two different places, so the script reports
three thresholding protocols. The choice is not cosmetic: it flips the sign of the apparent
gap, and only one variant passes the FPR≈q competence gate.

| variant | how τ is set | on-diag FPR (target q=1%) | what it measures |
|---|---|---|---|
| **static-insample** (the literal spec) | (1−q) quantile of head A's scores on A's **cal-split** benign, shipped fixed to B | **9.3%** — FAILS ≈q | in-sample optimism + threshold drift, conflated |
| **static-xfit** | same but 5-fold out-of-fold cal-benign scores (honest τ) | 0.0% (under, n-limited) | static threshold transfer, in-sample bias removed |
| **budget-enforced** (the protocol control papers run) | (1−q) quantile of head A's scores on B's **deployed** (test) benign stream | **1.4%** — holds ≈q on every cell | the **ranking** failure at matched audit cost |

The competence gate (per the session brief: "sanity-check on-diagonal FPR≈q before trusting
any off-diagonal number") **disqualifies static-insample**: its diagonal FPR is 9.3%, not 1%,
because head A is trained on the very benign split its threshold is read from, so the (1−q)
quantile of training scores sits too low on held-out benign. Its off-diagonal FPR averages
**43.5%** (range 0–100%), so its "off-diagonal TPR 49% > diagonal 29%" is not a safety gain;
it is the artifact of auditing up to 100% of benign traffic off-lineage. You cannot compare
catch rates measured at 1% vs 100% audit spend.

**The deployment-correct protocol is budget-enforced.** A blue team has fixed auditing
capacity (humans can review ~1% of submissions) and audits the most-suspicious actions in
the live stream. Because attacks are rare, the top-q% most-suspicious ≈ the q% of benign
traffic with the highest scores, so FPR = q **by construction on every family** (verified:
diag 1.4%, off-diag 1.4%, range 1.2–1.8% at q=1%). Holding audit spend identical across all
cells is exactly what isolates the *ranking* transfer failure: at matched FPR, a TPR
difference IS a ranking difference, which is why this maps directly onto the +0.127 AUROC
interaction.

## Headline result — budget-enforced (full transcript)

TPR matrix (realized catch rate %, rows = calibration family A, cols = eval family B), q=1%:

```
            gemma   llama  mistral   qwen      diag
  gemma       1.2     2.6     0.0      8.6       1.2
  llama       2.5     1.3    31.6      2.5       1.3
  mistral     0.0     2.6    77.6      1.2      77.6
  qwen        1.2     2.6    60.5      3.7       3.7
```

| q | mean DIAG TPR (in-lineage, "published") | mean OFF-DIAG TPR (realized on another lineage) | SAFETY DROP (diag − off) | TPR interaction (net of main effects) | catch-rate ratio |
|---|---|---|---|---|---|
| **1%** | **29.1%→21.0%** [17.7, 24.4] | **9.7%** [8.9, 13.2] | **+11.3 pts** [+6.5, +14.1] | **+8.5 pts** [+4.8, +10.6] | **2.2×** |
| 2% | 22.3% [20.5, 26.3] | 11.6% [10.2, 14.9] | +10.7 pts [+7.5, +14.6] | +8.0 pts [+5.6, +11.0] | 1.9× |
| 5% | 25.7% [23.3, 32.8] | 15.4% [13.9, 18.6] | +10.3 pts [+6.5, +16.8] | +7.7 pts [+4.9, +12.6] | 1.7× |

(The 29.1% diagonal in the q=1% row is the full-test point estimate; 21.0% is the
cluster-bootstrap mean — the gap between them is small-sample discreteness, see caveats.)
All CIs exclude zero. The drop is stable across audit budgets and the interaction term is
significantly positive at every q.

**Per-eval-family column view** (same attacks; only the monitor's calibration lineage changes),
q=1% — this is where the honesty lives:

| eval family B | own-lineage TPR | foreign-lineage mean TPR | own − foreign |
|---|---|---|---|
| gemma | 1.2% | 1.2% | +0.0 |
| llama | 1.3% | 2.6% | −1.3 (inverts) |
| **mistral** | **77.6%** | **30.7%** | **+46.9** |
| qwen | 3.7% | 4.1% | −0.4 |

The mean realized-safety gap is **carried by the mistral lineage**: mistral's own head catches
78% of its sabotage at a 1% budget, but foreign heads catch only 31% of the same attacks.
For gemma/qwen both rates sit near the q=1% floor (the monitor barely ranks these families at
all at this budget, in- or off-lineage), and for llama the own head is actually slightly worse
than foreign heads. This is the identical structure as the AUROC matrix (mistral diagonal 0.96
dominant; llama's own head not special) and the same caveat the Control-11 LoRA result carries.
The interaction term (+8.5) is the honest single-number summary because it is net of the
mistral row/column main effects, and the cluster-bootstrap-by-task CI accounts for the
concentration. Operationally the claim is precise: **for the lineages a family-conditioned
monitor can rank at deployment budgets, calibrating off-lineage roughly halves realized safety
at identical audit cost.**

## Robustness — difficulty-matched attacks (full transcript, budget-enforced)

Re-running with each eval family's test attacks subsampled to an identical main-task-success
composition (same machinery as `track2_difficulty.py`, so off-lineage attacks are
equally-successful by construction) leaves the gap essentially unchanged:

| q | diag | off | DROP | interaction |
|---|---|---|---|---|
| 1% | 20.6% | 9.4% | **+11.2 pts** [+6.7, +14.2] | +8.4 pts [+5.1, +10.6] |
| 2% | 22.1% | 11.5% | +10.5 pts [+7.1, +14.8] | +7.9 pts [+5.3, +11.1] |
| 5% | 25.7% | 15.6% | +10.2 pts [+5.7, +16.3] | +7.6 pts [+4.3, +12.3] |

So the realized-safety gap is not an artifact of families differing in how successfully they
sabotage — equally-successful off-lineage attacks are still caught at roughly half the rate.

## The static-threshold failure (secondary finding, also a kill-shot)

If instead of re-ranking the live stream you ship a fixed threshold calibrated on family A
(the natural thing to do if you cannot rank the deployed stream online), the protocol simply
**does not hold its audit budget off-lineage**:

- **static-insample, q=1%:** off-diagonal realized audit spend = 43.5% of benign traffic
  (range 0–100%). Shipping mistral's threshold to gemma flags 98–100% of gemma's benign
  traffic; shipping it the other way flags ~0%. The threshold is meaningless off-lineage.
- **static-xfit, q=1%:** with the in-sample bias removed the diagonal holds (FPR 0.0%, n-limited),
  but off-lineage audit spend still ranges 0–69% (mean 10.6%) against a 1% target.

The operating point τ is itself family-specific (a direct consequence of Control 2: benign
transcripts are 2.6× separable by family, so each family's score distribution sits in a
different place). This is the CLAUDE.md §1a "operational calibration transfer" failure made
concrete: a control eval that calibrates τ on one family and reports TPR@1%FPR is, off-lineage,
reporting a number at an audit cost it never actually pays. It is arguably the more alarming
operational failure, but it is a *budget-validity* failure, not a clean catch-rate gap, which
is why the budget-enforced number is the headline and this is the "and it's even worse" beat.

## Action-only (behavioral cut) — honest null at deployment budgets

On the action-only embeddings (code only, prose stripped), the budget-enforced realized-safety
gap is null: q=1% drop −1.3 pts [−3.9, +3.6], interaction −0.9 pts [−2.9, +2.7]; q=5% drop
+2.1 pts [−3.3, +5.6]. This is consistent with, not contrary to, the AUROC story: the
behavioral residual that survives in ranking terms is small (AUROC interaction +0.039), and a
+0.04 ranking edge does not convert into a measurable catch-rate gap at a 1–5% audit budget
with n=200 (TPR@1%FPR is dominated by the top one or two benign order-statistics at this n).
**The realized-safety gap reported here is carried by the full-transcript signal (stylistic +
behavioral combined).** We do not claim a behavioral-only deployment gap; the behavioral
transfer-failure component is a ranking-level (AUROC) finding, not a deployment-budget one at
this sample size. Scaling to paper-n (300–500 benign/family) is the way to test whether the
behavioral residual surfaces at a fixed budget.

## Online safety-game (assumption-dependent, clearly labelled)

A fuller AI-Control-style translation (Greenblatt et al.): i.i.d. tasks, rare attacks; auditing
an attack ends the game in the blue team's favor (shutdown-on-first-catch); an unaudited attack
succeeds with probability s. Long-run P(blue wins) = t / (t + (1−t)s), where t is the
budget-enforced TPR. At s=1 (every unaudited attempt is a successful backdoor) safety equals
the TPR; smaller s is a less capable red team. Budget-enforced, q=1%:

| s (per-attack unaudited success) | mean diag safety | mean off-diag safety | gap |
|---|---|---|---|
| 1.0 (conservative floor) | 21.0% | 9.7% | +11.3 pts |
| 0.5 | 24.9% | 14.1% | +10.8 pts |
| 0.2 | 30.7% | 21.7% | +9.0 pts |

The diagonal-vs-off-diagonal safety gap is ~9–11 points across the assumption range. This row
is explicitly assumption-dependent (attack frequency and per-attack success are stipulated, not
measured) and is presented as a sensitivity sketch, not a headline number.

## Why this is the deployment-facing kill shot

1. **It converts a metric objection into a guarantee objection.** A reviewer can argue about
   whether AUROC is the right monitor metric; they cannot argue that a protocol auditing 1% of
   traffic and catching 21% of in-lineage sabotage will, deployed on a different model lineage,
   catch only ~10% at the same cost. The safety guarantee a control eval publishes is the
   in-lineage diagonal; the realized guarantee on a new lineage is the off-diagonal, and it is
   roughly half.
2. **It pre-empts the "just re-threshold on deployment traffic" rebuttal.** The budget-enforced
   protocol *is* the re-thresholded one (FPR held at q on every family), and the gap survives —
   so the failure is in the monitor's *ranking* of a foreign family's sabotage, not merely in a
   misplaced operating point. The static-threshold variant then shows the operating point is
   broken too, so both layers of the protocol fail off-lineage.
3. **It is robust where it matters.** The gap holds under difficulty-matching (equally-successful
   off-lineage attacks are still caught at half the rate) and is the same diagonal-dominant
   structure, now in deployment units, as the +0.127 AUROC interaction and the +0.179 LoRA
   replication — so it inherits their robustness rather than standing alone.

## Reproduce

```
.venv/bin/python scripts/track2_safety.py track2 --xfit            # full transcript, 3 variants
.venv/bin/python scripts/track2_safety.py track2 --matched         # difficulty-matched robustness
.venv/bin/python scripts/track2_safety.py track2_actiononly --xfit # behavioral cut (null at budget)
```

Writes `results/safety_translation_{track2,track2_matched,track2_actiononly}.json`.

## Honest caveats

- **The mean gap is mistral-carried.** At q=1% three of four diagonal cells are near the audit
  floor (the monitor barely ranks gemma/llama/qwen at a 1% budget, in-lineage included), so the
  21%→10% gap is driven by the one lineage the monitor ranks well (mistral). Same concentration
  as the headline AUROC result; the interaction term and cluster bootstrap account for it, and
  the per-family column view above states it plainly. The clean claim is "for rankable lineages,
  off-lineage calibration ~halves realized safety," not "for every family."
- **Small-n discreteness.** At n≈55–81 test benign per family, the top-1% is one to two order
  statistics, so realized FPR lands at 1.2–1.8% rather than exactly 1.0% and the bootstrap mean
  TPR can sit a few points off the full-sample point estimate. Paper-n (300–500 benign/family)
  removes this.
- **One substrate (APPS), n=200, open-weight families.** Same external-validity caveats as the
  rest of the result; the bigcodebench replication and scale-n are the outstanding items.
- **Static-insample is reported but not trusted** (fails FPR≈q); budget-enforced is the safety
  number.
