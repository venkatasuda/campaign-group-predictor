# Predicting Profitable Customer Groups

**A decision-support system for direct marketing campaigns**

*Machine Learning and Engineering Challenge · August 2026*

> **Confidential — LP-Internal.** The dataset and the assignment brief are not reproduced in
> this document.

**Live system**

| | |
|---|---|
| Prediction API | `https://campaign-api-395867964283.europe-west3.run.app/docs` |
| Web interface | `https://campaign-frontend-htvxic2lsq-ey.a.run.app` |
| Repository | private — `github.com/venkatasuda/campaign-group-predictor` |

*Both services scale to zero. Steady-state latency is **79 ms median, 116 ms p95**
(`reports/latency.json`, 50 requests, 0 failures). The first request after an idle period is
slower — the container must start and deserialise a 47 MB model — and has been observed at
around 15 seconds on a genuinely cold instance, though that is not reproducible on demand.
See §7.*

---
---

# Part 1 — For the business reader

*Two pages. No statistics vocabulary. If you read nothing else, read this.*

---

## 1.1 The problem, in one paragraph

Every campaign round, the marketing team has two candidate customer groups and must pick one
to receive the budget. Today that choice is made before anyone knows which group will pay off,
and the answer only becomes clear after the money is spent. Sometimes neither group was worth
targeting at all.

**The question this system answers:** *given what we know about two groups before the
campaign, which should we target — or should we skip this round entirely?*

## 1.2 What the history shows

Looking at 6,620 past campaigns:

| Outcome | Share |
|---|---|
| Group 1 was the better choice | **46.5%** |
| Group 2 was the better choice | **28.4%** |
| **Neither group was profitable** | **25.2%** |

Two facts stand out, and both shape everything that follows.

**One campaign in four lost money no matter which group was chosen.** That is 1,667 rounds
where the only correct decision was *not to run the campaign*. No amount of better targeting
would have saved them. This makes "skip it" a real lever, not a technicality.

**The two positions are not evenly matched.** Group 1 wins roughly one and a half times as
often as group 2. So a rule as simple as *"always target group 1"* already succeeds **46.5%**
of the time. That is the bar. Any system that cannot beat it is not worth deploying.

## 1.3 What the system delivers

**Used on every campaign**, the model picks correctly **54.8%** of the time against the 46.5%
baseline — an improvement of **8.4 percentage points**. We are confident the true improvement
lies between **5.7 and 11.2 points**; it is not zero.

**Used only where it is confident, it does considerably better.** The system reports how sure
it is. Restricted to the campaigns where confidence is high:

| | Campaigns decided | Correct |
|---|---|---|
| Decide everything | 100% | 54.8% |
| **Only when confident** | **35%** | **71.6%** |

On those same confident campaigns, the "always group 1" rule gets **57.5%** right — so the
real advantage there is **14.1 percentage points**, not the 25 that a careless comparison
would suggest. Easy campaigns are easy for everyone; we compare like with like.

**The recommendation: automate the third of campaigns the system is confident about, and send
the rest to a person.** That is a smaller change to approve than "let the model decide
everything", and it is where the model is actually good.

## 1.4 What it cannot do yet

**It is poor at spotting doomed campaigns.** Of the campaigns that would have lost money
regardless, the system catches only **8 in 100**. This is the most valuable thing it could
do — a campaign correctly skipped saves the entire budget, whereas better targeting only
converts a loss into a gain when the other group would have paid off.

We know why: the information needed to make that call does not appear to be in the data we
have. Every test we ran points the same way.

**The 8.4-point figure is a measurement of better decisions, not a promise of revenue.** It
counts campaigns, and every campaign counts equally — but a large campaign and a small one do
not carry the same euros. Turning this into a money figure needs two things we do not have:
the actual cost of running a campaign, and a live test.

## 1.5 The two cheapest things that would help

Neither is a modelling task.

**1. Tell us what one column means.** Of the 67 pieces of information available before a
campaign, a single one carries more predictive signal than the other 66 combined. It is
anonymised, so nobody can say what it measures. One conversation with whoever owns the data
would likely be worth more than any further modelling work.

**2. Confirm what a campaign costs.** The system already weighs mistakes against each other —
spending on a doomed campaign is a different kind of error from skipping a good one. It
currently uses assumed figures. Real numbers would change the recommendation on a measurable
share of campaigns and cost one meeting.

## 1.6 What we recommend doing next

1. **Run a controlled test.** Split upcoming campaigns: half decided as today, half by the
   system, and compare actual returns. Roughly **200 campaigns per side** if restricted to
   the high-confidence subset, or 559 if applied to everything. The restricted version is
   cheaper and the effect is larger.
2. **Automate the confident third**, escalate the rest.
3. **Answer the two questions above** — they cost time, not money.
4. **Do not treat the remaining gap as a modelling problem.** Every diagnostic says the
   limit is the information available, not the choice of algorithm.

---
---

# Part 2 — Technical detail

*Pages 3–12. Methodology, evaluation, engineering.*

---

## 2. Data and its limits

### 2.1 Structure

6,620 rows × 71 columns. One row is one campaign — a comparison of two customer groups.

| Block | Columns | Count | Available before the campaign? |
|---|---|---|---|
| Group 1 characteristics | `g1_1` … `g1_20` | 20 | Yes |
| Group 2 characteristics | `g2_1` … `g2_20` | 20 | Yes |
| Comparison features | `c_1` … `c_27` | 27 | Yes |
| **Post-campaign** | `g1_21`, `g2_21`, `c_28` | 3 | **No** |
| Target | `target` | 1 | The label |

**67 usable features. 3 forbidden. 1 label.** All columns are anonymised, which forecloses
causal interpretation for the whole analysis.

**The target mixes two different tests.** Classes 1 and 2 are *relative* — which group did
better. Class 0 is *absolute* — neither cleared a profitability bar. So the label encodes a
threshold as well as a comparison, and the problem cannot be reduced to a clean binary
without discarding a quarter of the data and the only decision where spend is avoided rather
than redirected.

### 2.2 Quality

Zero nulls across 470,020 cells. Zero duplicated rows, checked two ways: fully duplicated
rows (a handling artefact) and rows duplicated on *features only*, which would cap achievable
accuracy because identical inputs with different labels are unlearnable.

**One tempting inference was wrong and is worth naming.** An early version of this analysis
argued: each round compared two different groups, therefore rows are independent. That does
not follow. The same customer group could appear across several rounds with different
partners — correlated outcomes, no duplicate rows. **There is no campaign or customer
identifier**, so independence cannot be tested. If rows are clustered, the effective sample
size is below 6,620 and every confidence interval in this report is too narrow. Unresolvable
without a key from the data owner.

### 2.3 The post-campaign columns — and why the standard test was the wrong one

**Hypothesis, stated before testing:** the brief says three columns were recorded *after* the
campaign, and separately that the company "evaluated the ROI of both groups" afterwards. So
those columns probably *are* that evaluation. If so, the target is nearly a deterministic
function of them and any model trained on them scores brilliantly and uselessly.

**Experiment.** Same model, identical folds, with and without.

| Feature set | CV macro F1 |
|---|---|
| With post-campaign columns (70) | 0.4948 |
| Pre-campaign only (67) | 0.4910 |

**Difference: +0.0038.** Essentially nothing. The hypothesis was falsified, and inspection
shows why: `g1_21` is continuous on [0, 1] with 6,289 distinct values; `g2_21` takes 52
discrete values over 2.5–19. Incommensurable scales — they cannot be two comparable ROI
figures.

**The columns are still excluded, and the argument had to change.**

> They are excluded because **they do not exist at prediction time** — not because they leak.

This distinguishes **target leakage** (correlation with the label) from **temporal or
availability leakage** (the value does not exist when the decision is made). The conventional
check — *"does including it inflate the score?"* — **would have cleared these columns**.
Availability is the stronger criterion and the one that determines whether a model works in
production.

Enforced in three independent places so it cannot be bypassed by accident: the training
split, the API's feature adapter, and the HTTP request schema. A unit test fails if any lets
one through.

---

## 3. Feature engineering — including what did not work

### 3.1 One pipeline object

```
PairwiseFeatureBuilder → SimpleImputer(median) → StandardScaler → Classifier
```

Fitted as a unit, serialised as a unit. The same fitted transformations that ran at training
run at serving, so there is no separate preprocessing script to drift out of sync — the
classic source of training/serving skew.

### 3.2 Pairwise differences

The label *is* a comparison, so differences are the natural representation: a tree needs many
axis-aligned splits to approximate `g1_7 − g2_7` but only one if handed the difference. The
builder constructs `g1_i − g2_i` and `g1_i / g2_i` for each of the 20 pairs, expanding **67
features to 107**.

**Measured honestly: 0.5699 with, 0.5702 without.** It adds nothing on this dataset. Reported
rather than assumed. Retained because the reasoning is sound and the cost is zero, but the
honest statement is that the data did not reward it.

### 3.3 Class imbalance — deliberately untouched

No resampling, no SMOTE. The imbalance (25/46/28) is mild and, more importantly, *real* — it
reflects how campaigns actually turn out. Resampling would optimise for a world that does not
exist and would distort the calibrated probabilities the decision layer consumes. The
minority-class problem is handled in **reporting** (per-class recall always shown) and in the
**cost matrix**, not by fabricating rows.

### 3.4 Group-swap symmetry: a free lunch, tested and refused

"Group 1" and "group 2" look like arbitrary labels. Mirror every row, flip the label 1↔2, and
the training data doubles for free.

**Two preconditions were tested. Both must hold.**

**Is the mirror well defined?** The group blocks exchange cleanly; the comparison block does
not, because its features come in three types that behave differently:

| Type | Correct mirror |
|---|---|
| Signed difference `g1 − g2` | **Negate** |
| Symmetric `\|g1 − g2\|` | **Leave alone** |
| Ratio `g1 / g2` | **Invert** — negating a positive ratio produces impossible rows |

Columns are anonymised, so type is inferred empirically and verified by a two-sample KS test.
Five features flagged direction-dependent: `c_12`, `c_19`, `c_20`, `c_21`, `c_22`.

**Are the positions exchangeable at all?** A KS test on each of the 20 paired variables:
**7 of 20 differ significantly.** Group 1 is systematically a *different kind of group*.

**Verdict: augmentation refused — in code, not as a warning.**

The refusal is correct twice over. The position bias is *genuine signal* — it is what makes
"always target group 1" a 46.5% baseline, and mirroring would have destroyed the dataset's
best prior. And a second reason emerged by accident: a model using only `c_2`, the most
predictive feature, has a symmetry violation rate of **89.35%** despite `c_2` never being
flagged as direction-dependent. **The diagnostic missed the most important comparison
feature**, so augmented rows would have carried an uncorrected `c_2`.

### 3.5 Comparison-block redundancy

**15 of 27 `c_` features are reconstructible** from the group blocks; the other 12 carry
information absent from them entirely. Two consequences: importance is *diluted* across
correlated copies, so no single feature looks as strong as it is — which is why §5.3 measures
blocks and counts directly rather than trusting one ranking; and the `c_` block cannot be
dismissed as derived bookkeeping.

---

## 4. Evaluation protocol

**This section is the one I would most want a reviewer to read**, because the project's own
protocol was found to be contaminated and corrected.

### 4.1 Three splits, not two

```
6,620 campaigns
├── Test:        1,324  (20%)   ← selected on ONCE, on the locked champion
└── Development: 5,296
    ├── Calibration: 1,060      ← calibrator, thresholds, gate chosen here
    └── Train:       4,236      ← fitting + cross-validation
```

The governing principle:

> **Anything chosen by looking at data must be chosen on data that is neither the fitting
> data nor the final evaluation data.**

Three things are chosen after fitting — whether to keep a calibrator, a decline threshold, a
confidence gate. With only train/test there is nowhere to make those choices except the test
set.

**Two seeds, deliberately separate.** `--random-state 42` controls model initialisation and
fold assignment; `--holdout-seed 20260819` controls *which campaigns* are held out.
Conflating them is how "let me try another seed" quietly becomes a new test set.

### 4.2 The contamination, and the fix

An earlier version of this project scored **every candidate on the test set** "for the
leaderboard" while selecting on cross-validation. The intent was benign. The effect was not:

> The test figures were on screen while the decision was being made. **Selection bias needs
> only visibility, not intent.**

Worse, the report claimed the test set was evaluated once while the code touched it seven
times. It was found by an external reviewer, not by me.

Auditing for the same pattern found it in four places:

| Procedure | Was selected on | Now selected on |
|---|---|---|
| Model choice | CV, with test scores visible | CV only; test unreferenced in the loop |
| Calibrator, keep or drop | The set it was fitted on | A disjoint half of the calibration split |
| Decline threshold τ | Test | Calibration split |
| Confidence gate | Test | Calibration split, then frozen |

**The guarantee is now a mechanism.** `test_only_the_champion_is_evaluated_on_the_test_set`
fails if `predict(x_test)` re-enters the candidate loop.

### 4.3 The precise claim, because the loose one is false

After the champion is frozen the test set is **read** several times — metrics, symmetry,
calibration reporting, decision policy, drift. Grep `x_test` in `train.py` and you find about
eight hits. That is not contamination: reading a frozen model's predictions to compute a
different quantity changes nothing, because **no choice depends on the result**.

What happens once is the **selection**. "Evaluated once" is the stronger claim and it is
untrue.

### 4.4 The holdout is not virgin, and that is stated

Earlier phases cross-validated over the full dataset and ran target-aware exploration on it.
Re-splitting under a new seed reshuffles rows the project has already collectively seen — **no
seed change manufactures unseen data.** The honest description is a **reused out-of-sample
confirmation set**, and its figures should be read as confirmation rather than as an unbiased
estimate of future performance.

---

## 5. Modelling

### 5.1 Seven candidates, four families

| Model | CV accuracy | Class weighting |
|---|---|---|
| **Random forest (champion)** | **0.5781 ± 0.0051** | `balanced_subsample` |
| XGBoost | 0.5741 ± 0.0090 | none |
| CatBoost | 0.5656 ± 0.0088 | `Balanced` |
| HistGradientBoosting | 0.5649 ± 0.0067 | none |
| MLP | 0.5626 ± 0.0183 | none |
| LightGBM | 0.5590 ± 0.0124 | `balanced` |
| Logistic regression | 0.5229 ± 0.0181 | `balanced` |

**A confound worth disclosing.** The third column is not uniform: four candidates carry a
class-weighting policy and three do not, because each was configured with its family's
conventional default rather than to a single stated policy. **This leaderboard therefore
compares model families *and* weighting policies at the same time**, and cannot separate
them. Given that the top five sit within 1.6 points and the champion's margin is inside
noise, it is not a confound that changes the conclusion — but it is one a reader should not
have to discover. The correct comparison runs each leading family both with and without the
same business-motivated weighting.

**The important observation is not who won.** Random forest leads by 0.40 points against a
combined fold spread of 0.014 — that is not a separation. Three independent lines confirm it:

- **Nested cross-validation ranks them in the opposite order** (XGBoost 0.5899 ± 0.0026,
  RF 0.5816 ± 0.0095)
- **Two hyperparameter searches over identical folds selected different champions**
- **Changing only the split seed moved test accuracy by 2.72 points** — 0.5483 at seed
  20260819, 0.5755 at seed 42. The standard error on 1,324 rows is 1.37 pp, so that is two
  standard errors of ordinary sampling variation

> **The candidates differ by 0.4 points. The measuring instrument moves by 2.7.**

**So the defensible property is not that the best model was found.** It is that the choice was
made by a stated rule, before the test set was opened, with no human override.

### 5.2 The selection metric

**Accuracy, because accuracy *is* the campaign success rate** — the share of campaigns where
the chosen action matches the profitable outcome. It is the only candidate metric that is
directly the business objective.

**Macro F1 was tried first and rejected.** It weights the three classes equally, rewarding a
model that lifts minority recall by sacrificing the dominant class — and it selected a model
whose success rate fell **below the naive baseline**.

**Business lift is deliberately not the selection metric**, despite being the headline. Lift
is measured on test; selecting on it would be selecting on test. Under cross-validation
accuracy ranks models identically anyway, since lift is accuracy minus a constant.

Everything else is still reported — macro F1, balanced accuracy, per-class figures, confusion
matrix. Accuracy alone would hide that class-0 recall is 8.4%.

### 5.3 Held-out performance

| Metric | Value |
|---|---|
| **Accuracy (campaign success rate)** | **0.5483** |
| Balanced accuracy | 0.4780 |
| Macro F1 | 0.4482 |
| Expected calibration error | 0.0213 |

| Class | Precision | Recall | Support |
|---|---|---|---|
| 0 — neither profitable | 0.257 | **0.084** | 333 |
| 1 — group 1 | 0.615 | 0.797 | 615 |
| 2 — group 2 | 0.498 | 0.553 | 376 |

```
                 predicted
              0     1     2
actual  0    28   184   121     ← 333 unprofitable, 28 caught
        1    36   490    89
        2    45   123   208
```

### 5.4 Calibration — measured, then rejected

The decision layer multiplies probabilities by monetary amounts, so a stated 61% must mean
61%.

**Protocol: three disjoint sets.** Model fitted on train; calibrator fitted on half the
calibration split (530 rows); the decision to keep it made on the *other* half. Two halves
rather than one, because **a calibrator scored on its own fitting data always looks well
calibrated.**

**Result: isotonic reduced ECE from 0.0530 to 0.0498 — and was rejected.** That is a 6.1%
relative improvement against a **10% requirement stated in advance**.

The margin exists because "apply whichever scores lowest" is not a decision rule: on 530 rows
ECE has a standard error comparable to the differences being compared, so *something* always
wins and the rule always fires. An earlier version had no margin and would have shipped this
calibrator. The uncalibrated model scores **ECE 0.0213** on test — already well calibrated.

### 5.5 What the model is made of

Permutation importance and SHAP, **both computed on the calibration split**, because a ranking
derived from test labels becomes selection-on-test the moment anything downstream uses it —
and §5.6 does.

They complement rather than duplicate: permutation importance asks *how much does the score
drop if this column is destroyed?*; SHAP asks *how much did this feature move this
prediction?*, which is what a campaign manager asking "why this group?" needs. They **disagree
informatively** — permutation importance splits credit between correlated copies, and 15 of 27
comparison features are reconstructible, so a strong feature can look weak. Agreement is
evidence; disagreement points at collinearity.

Both rank **`c_2` first.**

### 5.6 The finding that reframes the project

| Model | Features | Test accuracy | Class-0 recall |
|---|---|---|---|
| XGBoost | `c_2` alone | **0.5650** | **0.156** |
| XGBoost | all 67 | 0.5559 | 0.132 |
| Random forest | `c_2` alone | **0.5642** | **0.156** |
| Random forest | all 67 | 0.5483 | 0.084 |

**One feature beats sixty-seven, in two model families** — so it is a property of the data,
not a quirk of one algorithm. And it nearly **doubles class-0 recall**, the capability the
deployed model is worst at.

**Block ablation agrees:**

| Block | Features | CV |
|---|---|---|
| `c_2` alone | 1 | **0.6013** |
| Comparison block only | 27 | 0.5876 |
| All features | 67 | 0.5781 |
| Group blocks only | 40 | 0.5208 |

**The 40 group-characteristic columns cost 5.7 points.** Removing them improves the model.

**One claim deliberately not made.** The feature-count sweep is **not monotonic** — 1 →
0.6013, 3 → 0.5413, 5 → 0.5751, 40 → 0.5822. The middle of that curve is noise. Only the
endpoint comparison is claimed, and it holds on test.

**And what is *not* concluded:** that the other 66 features carry no information. They could
be redundant, individually weak but jointly useful, or useful only under distribution shift.
The defensible claim is **limited incremental predictive value in this evaluation**.

---

### 5.7 Two questions a stakeholder asks next

**Would more campaigns help? Diminishing returns.** The final doubling of training data bought
**0.0097** accuracy. The curve has not flattened, but the slope is shallow enough that
collecting more campaigns is unlikely to be the best available investment. It does not follow
that more data *cannot* help — only that the return per row is now small, which is consistent
with everything else in this section pointing at the features rather than the sample size.

**Would combining models help? Unresolved — and reported as such.** Stacking scored above the
best single base, but with a fold-to-fold standard deviation of **0.0006** across three folds,
which is implausibly tight, and using reduced model configurations that do not match the
leaderboard in §5.1. Two of the four bases also carry class weighting while two do not (see
§5.1), so the comparison confounds ensembling with weighting policy.

Reported as unresolved rather than as a positive result. Maintaining four models to serve one
prediction needs clearer evidence than a suspiciously small standard deviation on a
non-comparable configuration.

---

## 6. Business impact and validation

### 6.1 Four baselines

| Strategy | Success rate |
|---|---|
| Run no campaigns at all | 25.15% |
| Always target group 2 | 28.40% |
| Random choice between groups | 37.42% |
| **Always target group 1** (best naive) | **46.45%** |
| **Model** | **54.83%** |

**Lift: +8.38 pp**, 95% bootstrap CI **[+5.66, +11.18]** over 1,000 resamples (SE 1.38 pp),
excluding zero. The bootstrap resamples test rows and recomputes **both** the model's rate and
the baseline's on the same resample, accounting for the two moving together.

*"Run no campaigns at all" is included because it is the baseline easiest to forget and can be
the strongest — if most campaigns were unprofitable, doing nothing beats every targeting rule.*

### 6.2 The operating point

Threshold selected on the calibration split by a stated rule — *best accuracy subject to
coverage ≥ 30%* — then frozen and applied to test **once**.

| | Coverage | Model | Baseline, **same** campaigns | Lift |
|---|---|---|---|---|
| Decide everything | 100% (n=1,324) | 54.83% | 46.45% | +8.38 pp |
| **Confidence ≥ 0.60** | **34.8%** (n=461) | **71.58%** | **57.48%** | **+14.10 pp** |

Selected at 73.9% on calibration, delivered 71.58% on test — **it generalised.** The fourth
column is essential: quoting 71.58% against the baseline's overall 46.45% compares a gated
model to an ungated baseline.

### 6.3 The decision layer

`argmax` answers *what is most likely*; the business asks *what action maximises expected
return*. Those differ when mistakes cost different amounts.

| True state ↓ / Action → | do not run | target group 1 | target group 2 |
|---|---|---|---|
| **neither profitable** | **0.0** | +1.0 | +1.0 |
| **group 1 profitable** | +0.5 | **−1.0** | +1.0 |
| **group 2 profitable** | +0.5 | +1.0 | **−1.0** |

The system takes the minimum-expected-cost action, differing from `argmax` on **38 of 1,324**
campaigns. Near-ties — within 0.05 in expected cost — are flagged `review_required` and routed
to a person: **7.55%** of campaigns.

**The costs are assumptions and this is stated loudly.** The dataset contains no monetary
values, so a single ROI figure cannot honestly be derived. Results are reported as a
**sensitivity analysis across three ratios**, not one invented number.

### 6.4 The A/B test

Randomised at campaign level. Control = current process. Treatment = model recommendation,
skip when class 0 predicted. Primary metric **ROI per campaign**; guardrails on total revenue,
campaigns skipped, per-segment fairness.

```
n per arm = 2 · (z₁₋α/₂ + z₁₋β)² · p̄(1 − p̄) / (p₁ − p₂)²
p₁ = 0.4645, p₂ = 0.5483 → p̄ = 0.5064, effect = 0.0838
n ≈ 559 per arm
```

**559 per arm, 1,118 total** — a material fraction of the annual programme, worth saying
before anyone commits. **Restricting to the high-confidence subset raises the effect to
+14.10 pp**, and required *n* scales with the inverse square of the effect, so the requirement
falls to roughly **200 per arm**. It also changes behaviour on a third of traffic rather than
all of it: a smaller thing to approve.

Horizon fixed in advance; alpha spending if interim looks are needed.

### 6.5 Learning from deployed decisions

Once deployed the model determines which campaigns run, so its own data is not a random
sample. **Censoring** — a "do not run" means the outcome is never observed. **Confounding** —
among campaigns that do run, the ones the model liked dominate, so training naively teaches
the next model to agree with the current one rather than with reality.

This is a partial-feedback problem and cannot be repaired retrospectively; the logging must be
designed before launch. Required per decision: `decision_id`, features, calibrated
probabilities, action taken, **propensity**, exploration flag, model version, dataset hash.

**`exploration_rate` cannot be zero if inverse-propensity evaluation is wanted** — with no
exploration, propensities for unchosen actions are zero and the estimator is undefined.
**The shipped default is nevertheless 0.0**, because exploration spends real budget on
knowingly sub-optimal campaigns, and a config default should not commit a marketing team to
misallocating one campaign in twenty.

---

## 7. Engineering

### 7.1 Architecture

Two Cloud Run services in `europe-west3`, both scale-to-zero.

```
Streamlit UI ──┐
               ├──► FastAPI ──► Pydantic schema ──► FeatureTransformer (Adapter)
API clients ───┘                                          │
                                                          ▼
                                              ModelRegistry (loaded once)
                                                          │
                                                          ▼
                                     sklearn Pipeline (in-image, 47 MB)
                                     pairwise → impute → scale → RF
                                                          │
                                                          ▼
                                     DecisionPolicy + confidence gate
                                                          │
                                    Cloud Logging ◄───────┘
```

Diagrams, including a full sequence diagram, are in `docs/architecture.md` with a status table
marking each component **Deployed / Partial / Proposed**. Two are honestly marked partial: the
model is baked into the image rather than loaded from Cloud Storage, and the drift reference is
captured but nothing computes PSI at runtime.

**Deliberately absent:** a feature store (all 67 features arrive in the request body — nothing
is looked up at inference), a message queue (synchronous request/response fits campaign
planning), Kubernetes (Cloud Run is the right size for one model).

### 7.2 Design patterns

| Pattern | Where | What it buys |
|---|---|---|
| **Strategy** | `BasePredictor` ABC → two implementations | Swap the model without touching the API |
| **Factory** | `src/factory.py` | Construct from a config string; adding a model type touches one file |
| **Registry** | `ModelRegistry` | One load at startup; the single seam for moving to Cloud Storage |
| **Adapter** | `FeatureTransformer` | Nested JSON → the exact 67-column frame; also blocks leakage a second time |
| **Pipeline** | sklearn | Preprocessing + model as one serialisable object |
| **Dependency injection** | FastAPI `Depends()` | Tests inject stubs; no global monkeypatching |

### 7.3 Input validation

The schema rejects, with `422` and a reason naming the field:

| Input | Why |
|---|---|
| Missing, unexpected, or post-campaign keys | The contract is exact |
| `NaN` | Indistinguishable from `null` downstream but arrives by a different path — usually a failed upstream computation. A caller should have one *deliberate* way to say "no value" |
| `±inf` or `\|value\| > 1e6` | Finite-but-implausible values pass every type check and overflow during standardisation; the caller then gets a `500` for a client error |
| More than 20% of features `null` | The median imputer fills every gap confidently, so 67 nulls previously returned a well-formed prediction with a confidence score, built entirely from training medians. **A campaign must never be approved from an empty request** |

Verifiable against the live service. These are correctness properties, not perimeter ones,
which is why they sit in the application rather than behind a gateway — no amount of
authentication stops an authorised caller submitting an empty payload.

### 7.4 Testing

**427 tests across 17 modules, 91.43% branch coverage** against an enforced `fail_under = 90`
in `pyproject.toml`, so the gate behaves identically locally and in CI.

Three tests assert *properties* rather than return values:

- `test_only_the_champion_is_evaluated_on_the_test_set` — fails if the test set re-enters the
  candidate loop
- `test_exploration_works_across_separate_single_row_calls` — the service handles one campaign
  per request; every earlier exploration test used a batch, which is why a bug that disabled
  exploration entirely for single requests survived
- `TestUnhandledExceptions` — asserts a `500` carries `X-Request-ID` and does *not* echo the
  underlying exception message

`train.py` is the weakest module at **72%**; the untested paths are the calibration branch and
MLflow logging, which need a full training run. It is included in the measurement — an earlier
version excluded it, which made the number flattering.

The suite runs on **synthetic fixtures with the real column contract**. That is not only
convenience: the dataset is never committed, so **CI has nothing to train on by design**, and
the fixtures are what make the code verifiable by a machine not permitted to see the data.

### 7.5 CI/CD and infrastructure

`.github/workflows/ci.yml`, seven jobs: **data-guard** (asserts no dataset in the tree *and*
none in git history, against an exact three-file allow-list rather than a directory exemption
— runs first and alone), **lint** (ruff, black, mypy), **test** (coverage floor, optional
libraries installed so guarded paths are exercised), **terraform** (fmt, init without a
backend, validate), **security** (pip-audit against the *serving* requirements as
release-blocking, plus a non-blocking development audit, and bandit), **docker** (builds and
boots the degraded application), and **docker-with-model** (builds a synthetic artifact,
bakes it in, and asserts `/ready` reaches 200, `/model/info` is not the baseline, a real
prediction returns a conformant body, and the container is not running as root).

**CI is complete. CD is not, and the distinction is stated rather than blurred.**
`deploy.yml` holds the deployment procedure — keyless Workload Identity Federation, an
explicit confirmation input, a fail-closed check that refuses to build an image around a
missing model — but it cannot run end to end. `artifacts/model.pkl` is gitignored, a fresh
checkout never contains it, and nothing in the workflow downloads it, so a dispatch stops at
that guard. Deployment today is manual, from a controlled local environment, because the
approved model does not live in Git.

That is a scope decision, not an omission: committing a 47 MB binary to make the workflow
green would trade a real problem for a worse one. The missing piece is an artifact source —
a versioned GCS bucket or a registry — and the fetch step marks exactly where it plugs in.

Manual also for a second reason. Campaign targeting allocates budget, so a deploy is a
decision someone makes, not a side effect of merging.

Terraform provisions both services, IAM and Artifact Registry, with least-privilege service
accounts. The 0.60 confidence gate is **declared in Terraform**, not passed on a command line,
because a value in a reviewed diff cannot be lost to an argument-quoting bug.

### 7.6 Lineage and reproducibility

Every artifact carries the **dataset SHA-256**, scikit-learn version, Python version, holdout
seed and the evaluation-protocol string. `metrics.json` duplicates the library versions in
plain text, deliberately: reading them from the artifact requires unpickling it, which needs
the very libraries you are trying to identify.

Serving dependencies are **pinned exactly where they deserialise the pickle** (scikit-learn,
numpy, scipy, joblib) and left as ranges for the HTTP layer, which never touches the artifact —
pinning it would decline security patches for no safety gain. The container runs Python 3.12
to match the interpreter that produced the model.

`make train` carries the mandatory flags and `make verify-artifact` asserts the artifact on
disk is the documented run, exiting non-zero otherwise. `make deploy` depends on it.

---

## 8. Limitations

Stated plainly, because each one bounds a claim above.

1. **Class-0 recall is 8.4%** on 333 test campaigns — the widest uncertainty of any figure
   here, and the class with the clearest business value.
2. **The +8.38 pp lift is an offline estimate of decision quality, not confirmed ROI.**
   Accuracy weights every campaign equally; euros do not.
3. **Independence between campaigns cannot be verified** — no campaign or customer identifier
   exists. Every confidence interval here may be too narrow.
4. **No temporal validation is possible** — no time-ordering column, so generalisation to
   *future* campaigns rather than held-out ones is untested.
5. **The holdout is a reused confirmation set**, not a virgin test set.
6. **1,060 development rows are unused, and the evaluation set is smaller than it needs to
   be.** After calibration was rejected the model was not refit on train + calibration. The
   correct protocol is out-of-fold selection across all 5,296 development rows, then refit on
   all of them — 25% more training data, and an estimate averaged over four times as many
   evaluation rows.

   The gain is quantifiable rather than hypothetical. Accuracy measured on 1,324 test rows
   carries a standard error of 1.37 pp; over 5,296 out-of-fold rows that falls to roughly
   0.68 pp, which would approximately halve the width of the lift interval — from
   [+5.66, +11.18] to something nearer [+7.0, +9.8].

   **It was still not done, and "it would change every document" is not the reason.** The
   reason is that it would refine a number without changing a decision. The champion leads
   XGBoost by 0.40 points against a fold spread of 0.014 and a split-seed swing of 2.72
   points (§5.1); a tighter interval around a lift that is already significantly positive
   moves no recommendation — the model still ships, the policy still flags the same
   near-ties, and the class-0 weakness in limitation 1 is unaffected. Against a finite
   verification budget, one internally consistent set of numbers was judged worth more than a
   more precise set carrying the risk of partial reconciliation across eight documents, a
   deployment and a release archive.

   What would change the judgement: an out-of-fold estimate landing outside the current
   interval. That measurement is cheap — it needs no retrain and touches no artifact — and is
   the first thing to run before the next round of modelling work.
7. **53.8% of predictions fail to transform correctly under a group swap.** The symmetry
   requires 0→0, 1→2, 2→1, so for classes 1 and 2 *changing is correct* and staying the same
   is the violation — the figure measures failure to flip, not flipping. The positions are
   genuinely not
   exchangeable so this is defensible signal — but the model has partly learned *which slot* a
   group occupies and would degrade sharply and silently if the upstream convention changed.
   This is the monitoring signal I would expect to fire first.
8. **The cost matrix is assumed, not measured.**
9. **47 MB artifact and a slow cold start** against a **116 ms** warm p95. Cold start has
   been observed at ~15 s but is not reproducible on demand, so it is reported as an
   order-of-magnitude observation rather than a measurement. Three fixes exist; each
   changes the model or the artifact and therefore every reported number.
10. **No model registry** — the artifact is baked into the image. The seam exists; versioned
    retrieval from Cloud Storage is designed, not built.
11. **No authentication or rate limiting**, deliberately, so the service can be evaluated by
    opening a URL. `allow_unauthenticated = true` is a Terraform variable, not a hardcoded
    choice, and flipping it removes the `allUsers` invoker binding.

    Flipping it is **not sufficient on its own**, and saying only "production removes
    `--allow-unauthenticated`" would understate the work. The Terraform half is already
    there - the frontend has its own service account and holds `roles/run.invoker` on the API,
    which is the least-privilege binding a private deployment needs. The application half is
    not: `frontend/app.py` calls the API with a plain `requests.post` and no `Authorization`
    header, so making the API private today would break the frontend with 403s rather than
    secure it.

    The missing piece is small and specific - fetch a Google-signed ID token for the API's
    audience from the metadata server and attach it as a bearer token, refreshing on
    expiry - but it is missing, and a plan that omits it is a plan that fails on execution.
12. **Drift reference captured; nothing computes PSI at runtime.**

## 9. Next steps

Ordered by value per hour, and the top two are not modelling work.

1. **Recover what `c_2` measures.** One anonymised column carries more signal than the other
   66 combined and nobody can say why. One conversation.
2. **Confirm the campaign spend-to-margin ratio.** Changes the recommended action on a
   measurable share of campaigns. One meeting.
3. **Obtain a campaign or customer identifier** — makes independence testable and the
   intervals honest.
4. **Obtain a campaign date** — enables temporal validation, currently impossible.
5. **Adopt out-of-fold selection with a full refit** — recovers the 1,060 unused rows and
   removes the reused-holdout objection.
6. **Run the A/B test on the high-confidence subset**, measuring model against baseline on the
   *same* campaigns.
7. **Model "should we run this at all?" as a dedicated binary task.** Class 0 is where every
   candidate fails and where the money is.

---

## Appendix — five defects found, and what they have in common

Each is a case where **the visible output looked correct while something underneath was
wrong.** Those are the ones that survive review.

**1. Test-set visibility.** Every candidate was scored on test "for the leaderboard" while
selection ran on CV. Found by an external reviewer *after* the report claimed otherwise. Now
enforced by a test.

**2. Exploration silently dead in production.** The random generator was created *inside*
`decide_batch`, so every call replayed the first draw. A batch of 1,000 explored correctly;
1,000 single-row calls — what the service actually does — explored **zero** times. Every
existing test used batches.

**3. A right number with a wrong label.** A results table mapped the label `"mlp"` onto
whatever the champion was. When the champion changed the label did not follow. The metrics were
correct; the attribution was not.

**4. A shared estimator across two fits.** Fitting a one-feature and a 67-feature model from
the same estimator instance meant the second fit destroyed the first. The table stayed correct
— predictions were taken immediately after each fit — but the retained model was wrong, and it
only surfaced three steps later when something reused it.

**5. A green deploy serving nothing.** A malformed argument folded two environment variables
into one; `MODEL_PATH` pointed nowhere; a permissive fallback let the container start and serve
majority-class predictions through three consecutive "successful" deploys. The default is now
fail-closed.

**The through-line:** in every case the system produced plausible output while being wrong. The
only defence is a mechanism — an assertion, a test, a fail-closed default — not an intention.
That is the principle this project is built on, and it was learned the expensive way.
