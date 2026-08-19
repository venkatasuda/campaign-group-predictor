# Model Card — Campaign Group Predictor

*Confidential — LP-Internal.*

Figures below are taken from `reports/findings.json`, `reports/advanced_experiments.json`
and `artifacts/metrics.json`. Where a metric is not reproduced here, the artifact is the
source of truth rather than this document.

---

## 1. Model details

| Field | Value |
|---|---|
| Name | Campaign Group Predictor |
| Version | 1.0.0 |
| Date | August 2026 |
| Type | Multiclass classifier (3 classes) inside a scikit-learn `Pipeline` |
| Champion algorithm | **Random forest** (400 trees, `min_samples_leaf=2`, balanced class weights) |
| Champion selected by | **Cross-validated accuracy, automatically.** No human override was applied |
| Calibration | **None applied** — isotonic was fitted and then *rejected* by a stated rule; see §4 |
| Evaluation protocol | Candidates compared on cross-validation only; the test set **selected on once**, on the locked champion, after selection. It is *read* several times afterwards for reporting — see §4 |
| Status of the holdout | **Reused out-of-sample confirmation set, not a virgin test set.** Earlier phases cross-validated over the full dataset; a new split seed does not undo that |
| Split seed | `holdout_seed = 20260819`, separate from `random_state` |
| Owner | Venkata |
| Contact | See covering email |

**Architecture:** pairwise feature construction → median imputation → standardisation →
classifier → (optional, currently inactive) isotonic calibration → cost-sensitive decision
layer.

---

## 2. Intended use

**Primary use.** Given the pre-campaign characteristics of two candidate customer groups,
recommend which group a direct-marketing campaign should target, or recommend not running
the campaign at all.

**Primary users.** Campaign managers and marketing analysts, through the web UI or the
prediction API.

**Out of scope — do not use this model for:**

- Decisions about individual customers. Inputs are group-level aggregates; the model has
  no individual-level resolution and using it that way would be both wrong and a data-
  protection problem.
- Campaign types or channels not represented in the historical data.
- Forecasting the *size* of the return. The model ranks options; it does not estimate
  revenue.
- Any automated action without the decision layer's `review_required` flag being honoured.

---

## 3. Data

| Field | Value |
|---|---|
| Source | `customerGroups.csv` — historical campaign comparisons |
| Rows | 6,620 |
| Unit of observation | One campaign, comparing two customer groups |
| Features used | 67 (`g1_1..g1_20`, `g2_1..g2_20`, `c_1..c_27`) |
| Features excluded | 3 (`g1_21`, `g2_21`, `c_28`) |
| Target | 0 = neither profitable, 1 = group 1, 2 = group 2 |
| Class balance | 25.18% neither · 46.47% group 1 · 28.35% group 2 |
| Missing values | None (0.00%) |
| Duplicate rows | None |
| Integrity | SHA-256 recorded in the model artifact and in `metrics.json` |

**Excluded features — why.** `g1_21`, `g2_21` and `c_28` were recorded *after* the campaign
ran. They are strongly predictive and completely unavailable at prediction time; training
on them yields an excellent offline score and a useless production model. They are blocked
in three independent places (training split, feature adapter, HTTP schema).

**Feature semantics.** Columns are anonymised. No causal interpretation of feature
importances is possible, and none is claimed.

**Known data caveats**

- Outcomes exist only for campaigns the company chose to run — a selection effect.
- No counterfactual: for a class-1 campaign we never observe what group 2 would have
  returned.
- Data spans "many years", so drift is expected. **No time-ordering column was detected**,
  so a temporal split is impossible and generalisation to *future* campaigns — as opposed
  to held-out ones — is untested. Recovering a campaign date from the data owner would
  close this at no modelling cost.
- `c_` redundancy: **15 of 27** comparison features are reconstructible from the group
  blocks. The remaining 12 carry information absent from `g1_`/`g2_` entirely.
- **The two group positions are not exchangeable.** 7 of 20 paired variables differ
  significantly between the blocks, so group 1 is systematically a different kind of group
  from group 2. Symmetry augmentation was tested and refused on this evidence.

---

## 4. Performance

### How the split was used

| Split | Rows | Purpose |
|---|---|---|
| Train | 4,236 | Model fitting and cross-validation |
| Calibration | 1,060 | Calibrator fitting, threshold and gate selection |
| **Test** | **1,324** | **Selected on once, on the locked champion** |

**Two honest qualifications on that last row.**

*What "once" refers to.* After the champion is frozen the test set is **read** several times
— metrics, symmetry, calibration reporting, decision policy, drift — roughly eight times in
`train.py`. None of those reads feeds a choice, so none is contamination. What happens once
is the **selection**. "Evaluated once" is the looser claim and it is false; a reviewer who
greps `x_test` will find the hits and should.

*What the holdout actually is.* Earlier phases of this project cross-validated over the full
dataset and ran target-aware exploratory analysis on it. Re-splitting under
`holdout_seed=20260819` reshuffles rows the project has already collectively seen — **a new
seed does not manufacture unseen data**. These figures are therefore a *reused out-of-sample
confirmation*, not an unbiased estimate of future performance. The nested cross-validation
below is the better-founded generalisation estimate; the A/B test in the validation plan is
the only thing that settles it.

The candidate loop does not reference the test set at all. That is enforced by
`test_only_the_champion_is_evaluated_on_the_test_set`, which fails if a future change
reintroduces `predict(x_test)` into the loop — which is how the property was lost the first
time. An earlier version of this project scored all seven candidates on test "for the
leaderboard" while selecting on CV; the figures were visible while the decision was being
made, and selection bias needs only visibility, not intent.

### Selection — on cross-validation

| Model | CV accuracy |
|---|---|
| **Random forest (champion)** | **0.5781 ± 0.0051** |
| XGBoost | 0.5741 ± 0.0090 |
| CatBoost | 0.5656 ± 0.0088 |
| Hist gradient boosting | 0.5649 ± 0.0067 |
| MLP | 0.5626 ± 0.0183 |
| LightGBM | 0.5590 ± 0.0124 |
| Logistic regression | 0.5229 ± 0.0181 |

Random forest wins by 0.40 points over XGBoost against fold standard deviations of 0.005
and 0.009. **That is not a separation.** The champion is the automatic argmax of a stated
metric, chosen before the test set was opened — which is the defensible property here, not
the claim that it is the best model.

### Held-out performance

| Metric | Value |
|---|---|
| Accuracy (= campaign success rate) | **0.5483** |
| Balanced accuracy | 0.4780 |
| Macro F1 | 0.4482 |
| Weighted F1 | 0.5030 |
| Expected calibration error | 0.0213 |
| Best naive strategy | "always target group 1" — **0.4645** |
| Absolute lift | **+8.38 pp**, 95% bootstrap CI **[+5.66, +11.18]** over 1,000 resamples (SE 1.38 pp) |
| Relative improvement | +18.0% |

**Per class** — more informative than the headline:

| Class | Precision | Recall | Support |
|---|---|---|---|
| 0 — neither profitable | 0.257 | **0.084** | 333 |
| 1 — group 1 | 0.615 | 0.797 | 615 |
| 2 — group 2 | 0.498 | 0.553 | 376 |

**Class 0 is the weak point and it is the commercially interesting class.** A quarter of
campaigns are unprofitable for both groups, and that is the only outcome where spend can be
avoided entirely; the model recovers 8.4% of them. The decision layer nevertheless declined
**28** campaigns that were genuinely unprofitable. A "decline if P(class 0) ≥ τ" rule was
tested with τ selected on the calibration split by a stated rule: **no threshold improved on
plain `argmax`** while staying within 1% of its accuracy — a negative result about the
features, not about the decision rule.

### Calibration was rejected, not skipped

Isotonic regression was fitted on one half of the calibration split and evaluated on the
other (n=530) — three disjoint sets, so the calibrator is never selected on data it was fit
on. It reduced ECE from 0.0530 to 0.0498: a **6.1% relative improvement**, below the stated
**10%** threshold, so it was not applied.

The rule exists because an earlier version had no noise guard and would have shipped any
improvement, however small. On a 530-row selection split a 6% ECE movement is not
distinguishable from noise, and a rejected calibrator is a result rather than an omission.

### The honest generalisation estimate

A tuned model's best cross-validated score is a maximum over noisy estimates, so it is
biased upward. Nested cross-validation — five outer folds, each running a complete
independent hyperparameter search scored once on data that search never saw — quantifies it:

| Model | Nested CV accuracy |
|---|---|
| XGBoost | 0.5899 ± 0.0026 |
| Random forest | 0.5816 ± 0.0095 |

Roughly **1.2 points below** what a single tuned run reports; that gap is the optimism,
measured rather than caveated. Two qualifications: these runs used a larger training portion
than the final protocol (no calibration split), so the absolute values are not comparable to
the leaderboard above; and XGBoost leads here by 0.83 points, in the *opposite* direction to
the leaderboard. Taken with the champion flip observed between two search strategies, the
conclusion is consistent — **these models are not separable on this data.**

### When the model can be trusted

A confidence gate was selected on the calibration split by a stated rule (*best accuracy
subject to coverage ≥ 30%*), then frozen and applied to the test set once.

| | Coverage | Model | Baseline on the **same** campaigns | Lift |
|---|---|---|---|---|
| Decide everything | 100% (n=1,324) | 54.83% | 46.45% | +8.38 pp |
| **Confidence ≥ 0.60 (frozen)** | **34.8%** (n=461) | **71.58%** | **57.48%** | **+14.10 pp** |

The frozen threshold transferred to the holdout with **71.58% accuracy at 34.82% coverage**,
against **73.86% at 33.21% coverage** during calibration selection. The 2.28-point difference may
reflect ordinary sampling variation, threshold-selection optimism, or both; a single comparison
cannot separate them, and "the threshold generalised" would claim more than one observation
supports.

**The fourth column is the one that matters, and it was missing from an earlier version of
this document.** Quoting 71.58% against the baseline's *overall* 46.45% would compare a
gated model to an ungated baseline and inflate the gain. On the same 461 campaigns the
"always group 1" rule also does better than its average — 57.48%, because easy campaigns are
easy for everyone. The real gain at that operating point is **+14.10 pp**, not +25.

Two caveats remain even so. The subset is selected *by model confidence*, so it is not a
random sample of campaigns and the comparison is conditional on the model's own view of
difficulty. And the 54.83% full-coverage figure must always be quoted alongside, so the
operating point is never mistaken for headline accuracy.

The proposal is therefore *automate the confident third, escalate the rest* — with the
conditional lift stated, not the flattering one.

### Group-swap invariance

Which group is labelled "1" would be arbitrary if the positions were exchangeable. They are
not (§3), so the model legitimately uses position — but it uses it heavily:

**Read the direction carefully.** Exchanging the two groups should map the prediction
0→0, 1→2, 2→1. A *violation* is a failure to make that transformation — so for classes 1 and
2 **changing is the correct behaviour** and staying the same is the fault. Describing this
figure as "predictions that change under a swap" states the opposite of what is measured.

| | |
|---|---|
| Predictions that **fail to transform correctly** under a group swap | **53.8%** |
| Class-0 predictions that correctly stayed class 0 | 75.2% |
| Position bias in the labels | +18.11 pp |

**This is a deployment risk, not a robustness property.** The model has partly learned which
slot a group occupies rather than what the group is like. If the upstream convention for
assigning "group 1" ever changes, it will degrade sharply and silently.

**A related finding that undercuts the diagnostic itself.** A model using only `c_2` — the
single most predictive feature — has a symmetry violation rate of **89.35%** (89.43% under
XGBoost, so it is the feature, not the model), yet `c_2` was
*not* flagged as direction-dependent, so the mirror operation leaves it unchanged. The
diagnostic missed the most important comparison feature. Had symmetry augmentation been
enabled, every mirrored row would have carried an uncorrected `c_2`. This is a second,
independent reason the technique was refused, and it was found by accident rather than by
design.

---

## 5. Decision layer

The model outputs probabilities; the deployed system outputs an *action*. The action is
the one minimising expected cost under an explicit cost matrix, not the most likely class.

| True state ↓ / Action → | do not run | target group 1 | target group 2 |
|---|---|---|---|
| **neither profitable** | **0.0** | +1.0 | +1.0 |
| **group 1 profitable** | +0.5 | **−1.0** | +1.0 |
| **group 2 profitable** | +0.5 | +1.0 | **−1.0** |

Assumed values: campaign spend **1.0**, profit if correct **1.0**, opportunity weight
**0.5** — the last being the foregone margin from declining a campaign that would have paid
off, weighted below a cash loss because no money left the business.

**These are business inputs on a relative scale, not measured euros.** The dataset contains
no monetary values, so a single expected-ROI figure cannot honestly be derived from it.
Results are therefore reported as a **sensitivity analysis across three ratios** in
`reports/decision_sensitivity.csv` rather than as one invented number. Confirming the real
spend-to-margin ratio with the marketing team is the cheapest available improvement to this
system.

When the best and runner-up actions are within **0.05** of each other in expected cost, the
response is flagged `review_required` and routed to a human rather than automated. On the
test set this fires for **7.55%** of campaigns. The flag is part of the API response; the
service never suppresses it silently.

This is separate from, and complementary to, the confidence gate in §4: the review flag
catches *near-ties between actions*, the gate catches *low confidence overall*.

---

## 6. Ethical and fairness considerations

- **No individual-level data.** Inputs are group aggregates, which limits privacy exposure.
- **Feedback loops.** Once deployed, the model shapes which campaigns run, so future
  training data reflects its own choices. Retain a randomised holdout of campaigns that
  ignore the model, so unbiased data keeps arriving.
- **Fairness.** Features are anonymised, so protected attributes cannot be audited from
  the data alone. Before production, confirm with the data owner whether any feature
  proxies a protected characteristic. Track outcome rates by customer segment as a
  guardrail.
- **Automation bias.** The confidence and rationale are surfaced precisely so users do not
  treat the output as an oracle.

---

## 7. Limitations

1. **The +8.38 pp lift is an offline estimate of improved targeting decisions, not
   confirmed ROI.** It assumes the historical target is a faithful proxy for future
   profitability, that campaigns are independent, and that deploying the model does not
   change the population of campaigns proposed. Only the A/B test can convert it into a
   revenue claim.
2. Class 0 has 333 test rows and 8.4% recall; its estimate carries the widest uncertainty
   of any figure here.
3. No causal claims — the model supports the decision, it does not explain the market.
4. Performance will decay as customer behaviour changes; see the retraining trigger below.
5. The cost matrix is currently assumed, not measured.
6. **Independence between campaigns cannot be verified.** There is no campaign or customer
   identifier in the data, so if the same customer group appears in several rows the
   effective sample size is smaller than 6,620 and every confidence interval here is too
   narrow. This is unresolvable without a key from the data owner.
7. **The label's derivation is unknown, and that is the main known improvement path.**
   The post-campaign columns are *not* comparable per-group outcomes (`g1_21` is a
   continuous rate on [0,1]; `g2_21` takes 52 discrete values on a 2.5-19 scale) and are
   only weakly associated with the class label — so the profitability judgment encoded in
   the target was computed from information the dataset does not expose. Recovering that
   outcome definition from the data owner would enable a regression formulation with more
   signal than the 3-class label. See "Next steps" in the report.
8. **Test-set visibility was a real defect in this project's history, corrected in four
   places.** Candidate scoring, calibrator selection, threshold selection and the automation
   gate each once used, or could have used, the test set. All four now select on
   cross-validation or on the calibration split. The worst instance was found by an external
   reviewer *after* the report already claimed the test set was evaluated once — which is
   why the property is now asserted by a test rather than by a sentence.
9. **No temporal validation.** No time-ordering column was detected, so generalisation to
   *future* campaigns, as opposed to held-out ones, is untested.

---

## 8. Maintenance

| Item | Value |
|---|---|
| Retraining cadence | Monthly, plus on a drift alert |
| Drift monitoring | PSI on input features; predicted-class distribution |
| Retirement trigger | Campaign success rate falling below the naive "always group 1" rule (46.45%) on recent campaigns — at that point the model is subtracting value |
| Promotion rule | New candidate must beat the champion on **both** CV accuracy and business lift, **and** by more than one fold standard deviation. A 0.4-point win is not a win |
| Selection metric | **Accuracy**, because accuracy *is* the campaign success rate. Macro F1 weights the three classes equally and so optimises a different objective — a macro-F1 winner can have a lower success rate than the naive baseline |
| Experiment tracking | MLflow — every run logs parameters, metrics, artifacts, the dataset hash and a timestamp; nested runs per candidate |
| Rollback | Artifacts are versioned; rollback is a config change |

---
---

# ADR-001 — Choice of model family

**Status:** Accepted · **Date:** August 2026 · **Decider:** Venkata

## Context

We must choose a model to recommend which of two customer groups to target. The dataset is
small-to-medium, fully numeric, anonymised, with 67 features. The deployment target is
Cloud Run (CPU, scale-to-zero). Downstream, a cost-sensitive decision rule consumes the
predicted probabilities, so **probability quality matters at least as much as accuracy**.

## Options considered

**A. Logistic regression.** Interpretable, fast, trivially deployable. Risk: may underfit
if the boundary is non-linear in the pairwise differences.

**B. Gradient-boosted trees (CatBoost / LightGBM / XGBoost).** Long-standing strong default
for tabular data — robust to uninformative features (relevant: 67 anonymised columns, many
likely noise) and to non-smooth target functions. CatBoost is favoured in published
comparisons on log-loss and Brier score, i.e. exactly the probability-quality axis this
system depends on.

**C. Neural network (regularised MLP / TabM).** Credible recent evidence that well-tuned
regularised MLPs can match or beat boosted trees on tabular data. Costs more tuning effort
for an uncertain gain.

**D. Tabular foundation model (TabPFN-3, TabFM, TabICLv2).** As of mid-2026 these lead the
TabArena leaderboard by a wide margin on small datasets — TabPFN-3 at 1673 Elo versus 1433
for the best tuned tree. **However:** the TabPFN-3.0 licence permits evaluation and internal
benchmarking but prohibits commercial and production use, explicitly including *"using model
outputs as inputs to internal commercial decision-making"*. Campaign targeting is exactly
that. There is also evidence that such models lean on human-readable column metadata, which
this anonymised dataset lacks.

**E. AutoML (AutoGluon).** Strong stacked ensembles, but a heavy, opaque artifact that works
against the "any third person can understand it" requirement.

## Decision

**Train A, B and C under identical cross-validation folds and deploy whichever wins on
cross-validated accuracy, automatically.** D and E are reference benchmarks only — run,
reported, not deployed.

Two sub-decisions matter as much as the family choice:

- **Selection metric: accuracy, not macro F1.** Accuracy *is* the campaign success rate, so
  it is the only candidate metric that is directly the business objective. Macro F1 weights
  the three classes equally, which is a different objective and can prefer a model with a
  lower success rate.
- **No human override of the automatic winner.** Overriding on a secondary criterion after
  seeing held-out behaviour is selection on the evaluation set by another name.

## Outcome

The rule selected **random forest** (CV 0.5781 ± 0.0051), ahead of XGBoost at 0.5741 ±
0.0090 — a 0.40-point margin against fold standard deviations of 0.005 and 0.009. Nested
cross-validation puts XGBoost *ahead* by 0.83 points, and two search strategies over the
same families selected different champions.

**The models are not separable on this data.** This does not weaken the decision; it is the
reason the decision was made by a stated rule rather than by judgement. When candidates are
within noise, the defensible property is that the choice was made before the test set was
opened, not that the winner is genuinely best.

Isotonic calibration was fitted and then **rejected** — 6.1% relative ECE improvement
against a stated 10% threshold. The lead candidate anticipated in option B (CatBoost, chosen
partly for probability quality) placed third; the deployed random forest reached ECE 0.0213
uncalibrated, so the calibration step the option-B rationale depended on turned out not to
be needed.

## Rationale

1. The licence makes the highest-scoring model class (D) unusable for this purpose. This is
   a business constraint and it is decisive regardless of benchmark position.
2. Probability quality drives the decision layer, so both tree ensembles and a calibration
   step were in scope from the start — even though calibration was ultimately rejected on
   evidence.
3. A single joblib artifact deploys cleanly to Cloud Run on one vCPU
   (measured end to end from a developer machine: **p95 116 ms**, median 79 ms —
   `reports/latency.json`) and carries a straightforward permutation-importance explanation
   story.
4. Including a neural baseline and a foundation-model benchmark means the choice is made on
   measured evidence rather than on assumption.

## Consequences

**Positive:** legally deployable; fast, cheap CPU serving; explainable; reproducible;
strategies are swappable behind `BasePredictor` if the situation changes.

**Negative:** we knowingly deploy a model that scores below the benchmark leader, and one
whose margin over the runner-up is inside noise. Both gaps are measured and reported rather
than hidden.

**Revision note:** an earlier version of this ADR pre-committed to CatBoost and to macro-F1
selection. Both were changed by evidence — CatBoost placed third, and macro F1 was rejected
as a selection metric because it is not the business objective. The ADR is updated rather
than quietly rewritten, because the fact that a stated rule overruled a prior preference is
the point of having the rule.

**Follow-ups:** (a) verify whether Google's TabFM licence permits commercial use — if so,
re-open this decision; (b) if a TabPFN commercial licence is purchased, evaluate direct use
or teacher-student distillation — noting that published distillation gains concentrate on
datasets with ≤21 features and are marginal above that, and this dataset has 67.

## References

- TabPFN-3 Technical Report — arXiv:2605.13986
- Prior Labs model licensing — docs.priorlabs.ai/models
- TabArena: A Living Benchmark for Machine Learning on Tabular Data — arXiv:2506.16791
- Why do tree-based models still outperform deep learning on tabular data? — arXiv:2207.08815
- Pocket Foundation Models: Distilling TFMs into CPU-Ready Gradient-Boosted Trees — arXiv:2605.18654
- Post-tuning the decision threshold for cost-sensitive learning — scikit-learn documentation
