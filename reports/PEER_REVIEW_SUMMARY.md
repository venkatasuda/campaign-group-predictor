# Peer review — request, and what the answers turned out to be

*This document was circulated mid-project as a request for a second opinion. It is kept
here with the resolutions appended, because how the open questions were answered is more
informative than the questions were. Three of the four answers overturned what I had done.*

*No client data or materials are included here — this describes my own approach only.*

---

## The problem

A company runs marketing campaigns. Each round it has **two candidate customer groups**
and must pick one to target. Afterwards it learns which group would actually have been
more profitable — but by then the money is spent.

Build something that makes that choice **before** the spend.

**The data:** 6,620 past campaigns. For each one, 67 anonymised numeric features known
beforehand, and a 3-class outcome:

| Class | Meaning | Share |
|---|---|---|
| 1 | Group 1 was more profitable | 46.5% |
| 2 | Group 2 was more profitable | 28.4% |
| 0 | **Neither was profitable** | 25.2% |

That third class matters: a quarter of campaigns lost money regardless of targeting.

---

## What I did, step by step

### 1. Removed three columns that would have been cheating

Three columns were recorded *after* the campaign ran. Using them is like predicting a
football result from the final score — looks brilliant, useless in production. They're
now blocked in three places: the training split, the feature adapter, and the API schema.

Interesting wrinkle: they turned out to leak *barely at all* (+0.004 F1). So I kept the
exclusion but changed the argument — they're excluded because **they don't exist at
prediction time**, not because they leak. That reasoning survives the evidence; the
usual one doesn't.

This is temporal leakage — a question of *when a value becomes available* — rather than
target leakage, which is about correlation with the label. The distinction matters because
the correlation test came back nearly empty and would have talked me out of a correct
exclusion.

### 2. Tested my assumptions instead of applying techniques

I planned to use **symmetry augmentation** — since "group 1" and "group 2" look like
arbitrary labels, you can mirror each row and double your data.

Before doing it, I tested whether the two positions actually are interchangeable
(KS test per paired variable). **They aren't** — 7 of 20 paired variables differ
significantly. Group 1 is systematically a different kind of group.

So I dropped the technique: the two positions are not empirically exchangeable, and
mirroring cannot be justified without an additional invariance assumption the data does
not support.

*(A second, stronger reason emerged later — see the resolution to question 1.)*

### 3. Compared seven models across four families

Logistic regression, random forest, four gradient boosters, and a neural net — same
cross-validation folds throughout.

**They all landed within 2.5 accuracy points of each other, and the top five within 1.6.**
When completely different architectures converge like that, the ceiling isn't the algorithm.

### 4. Changed the selection metric after it picked a bad model

I originally selected on macro-F1. That chose a class-balanced model whose **campaign
success rate was worse than always picking group 1** — it maximised minority-class recall
by sacrificing the majority class.

Switched selection to accuracy, which *is* the success rate the business cares about.
Deliberately **not** to business lift, since lift is measured on the test set and
selecting on it would contaminate the final estimate.

### 5. Built a decision layer, not just a classifier

`argmax` answers "what's most likely". The business asks "what action maximises return" —
and those differ when mistakes cost different amounts. So the system states a cost matrix
and picks the minimum-expected-cost action, with a "too close to call" flag that routes
borderline campaigns to a human.

This is also why calibration matters: those probabilities get multiplied by money.

### 6. Deployed it

FastAPI on Cloud Run, a Streamlit interface, 427 tests across 17 modules at 91.43% behind a
enforced branch-coverage floor, containerised, CI, Terraform, MLflow tracking, drift-reference
statistics stored in the model artifact, and the dataset's SHA-256 recorded for lineage.

---

## Results, as they stand now

Champion **random forest**, selected automatically on cross-validated accuracy, scored
**once** on the test set after selection was locked.

| | |
|---|---|
| Best naive rule | always target group 1 — **46.45%** |
| Model | **54.83%** |
| Lift | **+8.38 pp**, 95% CI [+5.66, +11.18] |
| With a frozen confidence gate | **71.6%** on the **34.8%** of campaigns it is confident about |

**These are lower than the numbers this document originally carried** (56.7%, +10.3 pp,
76.0% at 31.5% coverage). The earlier figures came from a different champion selected under
a protocol the review found to be contaminated. The drop is the correction, not a
regression — see resolution 5.

---

## The four questions, and their answers

### 1. "The one-feature result — real finding, or a bug?"

*I described it as "clean and monotonic, which almost feels too tidy."*

**Half right, and my description of it was wrong.** Re-run on the shipped champion
configuration with the feature ranking derived from the calibration split rather than from
test:

| Features | CV accuracy |
|---|---|
| **1 (`c_2`)** | **0.6013** |
| 3 | 0.5413 |
| 5 | 0.5751 |
| 10 | 0.5788 |
| 40 | 0.5822 |
| 67 | 0.5758 |

It is **not monotonic** — there is a six-point collapse at three features and a recovery
afterwards. That shape is noise, and I should not have called it clean. **What survives is
the endpoint comparison**, and that does hold up on the test set: `c_2` alone reaches 0.5650
against 0.5559 for all 67 features, with class-0 recall almost doubled.

The block ablation agrees: the 27 comparison features alone beat all 67, and the 40
group-characteristic columns reach 0.5208 — 5.6 points above baseline, 5.7 below the full
set. Removing them *improves* the model.

I still decline to conclude the other 66 variables carry no information. They could be
redundant, individually weak but jointly useful, or useful only under distribution shift.
The defensible claim is **limited incremental predictive value in this evaluation**.

**The genuinely interesting part was an accident.** A `c_2`-only model has a symmetry
violation rate of **89.35%** (89.43% under XGBoost — the feature, not the model) — yet `c_2`
was never flagged as direction-dependent, so the
mirror operation leaves it unchanged and such a model cannot flip 1↔2. It nevertheless
reaches class-2 recall of 0.56, so `c_2` plainly does encode direction. **My symmetry
diagnostic failed on the single most predictive comparison feature.** Had I enabled
augmentation, every mirrored row would have carried an uncorrected `c_2`. That is a second
and much stronger reason the technique would have corrupted the data — and I found it by
accident, not by design.

### 2. "Champion choice — defensible, or am I rationalising?"

*I had shipped the second-place model because the winner's decision layer declined only 2
campaigns.*

**Rationalising.** The reasoning was: random forest wins CV accuracy but recovers almost no
class-0 campaigns, so ship the runner-up which recovers more.

The problem is *where the class-0 evidence came from*. It was held-out behaviour I had
already looked at. Overriding an automatic winner on a criterion measured after seeing
held-out results is selection on the evaluation set wearing a different hat — and it makes
the reported lift optimistic in a way no confidence interval captures.

**Now: the automatic winner ships, with no override.** Random forest, CV accuracy 0.5781 ±
0.0051. The class-0 concern was real, so it is addressed by a mechanism instead of by a
thumb on the scale — a "decline if P(class 0) ≥ τ" rule with τ chosen on the calibration
split by a stated rule. **No threshold beat plain `argmax`** while staying within 1% of its
accuracy. That is a negative result about the features, and reporting it is more useful than
having quietly swapped models.

Worth stating plainly: random forest's 0.40-point margin over XGBoost is inside noise, and
nested CV ranks them in the *opposite* order. **These models are not separable on this
data.** The defensible property is not that the best model was found — it is that selection
happened by a stated rule before the test set was opened.

### 3. "Selection metric — accuracy over macro-F1. Reasonable?"

**This one held up, and it is the only one of the four that did.** Accuracy *is* the
campaign success rate, so it is the only candidate metric that is directly the business
objective. Macro F1 weights three classes equally, which is a different goal.

The concern — that abandoning macro F1 hides something on the minority class — is correct
and is handled by reporting rather than by the metric. Macro F1 (0.4482), balanced accuracy
(0.4780) and per-class recall are all reported alongside accuracy, precisely so that 8.4%
class-0 recall cannot hide behind a 54.83% headline.

### 4. "Framing — does 'automate 41% at 72%' read as spin?"

**Partly, and the fix was to add a comparison I had left out.**

The gate itself is sound: the threshold is a fitted parameter, chosen on the calibration
split by a stated rule (*best accuracy subject to coverage ≥ 30%*), frozen, then applied to
test once. It was selected at 73.9% accuracy on calibration and delivered 71.6% on test, so
it generalised. The full-coverage 54.83% is always quoted alongside it.

What was missing: **the naive baseline would also score above its 46.45% average on that
same high-confidence subset.** Easy campaigns are easy for everyone. Comparing a gated model
against an ungated baseline overstates the gain, and I was doing exactly that. The
model-versus-baseline comparison on identical campaigns is now named as something the A/B
test must measure rather than something inferred here.

With that caveat attached, "automate the confident third, escalate the rest" is a fair
description of a deployable system rather than spin.

### 5. The finding nobody was asked about — and the most important one

A reviewer noticed that the report claimed *"the test set is evaluated once"* while the
training code scored **every candidate** on the test set inside the selection loop.

The intent was benign — a leaderboard for the write-up, with selection done on CV. The
effect was not. **The test figures were on screen while the decision was being made, which
is all selection bias requires.** It needs visibility, not intent. I had written the
guarantee into the report and then not implemented it, which is worse than not claiming it,
because a stated guarantee stops a reader from checking.

Auditing for the same pattern found it in four places:

| Procedure | Was selected on | Now selected on |
|---|---|---|
| Model choice | CV, but with test scores visible | CV only; test unreferenced in the loop |
| Calibrator (keep or drop) | The set it was fitted on | A disjoint half of the calibration split |
| Decline threshold τ | Test | Calibration split |
| Confidence gate | Test | Calibration split, then frozen |

Three fixes followed:

1. The candidate loop contains no reference to `x_test`. Only the locked champion is scored,
   in a later phase.
2. A test — `test_only_the_champion_is_evaluated_on_the_test_set` — fails if that changes.
   **The guarantee is now a mechanism rather than a sentence.**
3. Every selection rule carries a stated minimum improvement. Calibration applies only if ECE
   falls by at least 10%; isotonic delivered 6.1% and was rejected. Without a margin, a rule
   accepts any improvement including one indistinguishable from noise.

The honest summary of this project's evaluation history is that its most valuable finding was
a defect in its own protocol, and that an outside reader found it before I did.

---

## What I would still want a second opinion on

- Whether the `c_2` result warrants restructuring the model around a handful of comparison
  features, or whether the endpoint gap is small enough to leave alone.
- Whether Bradley–Terry–Davidson — which models order effects and ties explicitly — is worth
  the reformulation cost, given that position dependence is currently a documented risk
  rather than something the model structure handles.
- Whether declining to convert +8.38 pp into a euro figure is right, or whether a stated
  sensitivity range would be more useful to a business reader than a refusal.
