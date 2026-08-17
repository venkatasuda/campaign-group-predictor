# Business Impact and Validation Plan

Answers **ML Challenge question 3**: how much the model could improve the campaign
success rate, and how that improvement would be validated.

All figures below come from `artifacts/metrics.json` and `reports/findings.json`. Where a
number appears in both, the artifact is the source of truth rather than this document.

---

## 1. Defining "success"

A campaign decision is **successful** when the action taken matches the historically
profitable outcome:

| True class | Correct action | Value created |
|---|---|---|
| 1 | Target group 1 | Revenue from the profitable group |
| 2 | Target group 2 | Revenue from the profitable group |
| 0 | Run no campaign | Budget saved — no spend on an unprofitable campaign |

Class 0 matters: today an unprofitable campaign is only discovered *after* the money is
spent. A model that reliably flags class 0 converts a loss into a saving, which is
often the largest single source of ROI improvement.

## 2. Baselines the model must beat

| Strategy | Success rate | Note |
|---|---|---|
| Random choice between the two groups | `0.5 x (P(class 1) + P(class 2))` | Never declines a campaign |
| Always target group 1 | `P(class 1)` | |
| Always target group 2 | `P(class 2)` | |
| **Model** | share of test campaigns predicted correctly | |

Computed by `estimate_business_lift()` in `src/training/evaluation.py`.

**Result — champion random forest, held-out test set (n = 1,324):**

- Best naive strategy: **"always target group 1"** at **46.45%**
- Model: **54.83%**
- **Absolute lift: +8.38 percentage points** (relative +18.0%), 95% bootstrap CI over
  1,000 resamples: **[+5.66, +11.18] pp** — excludes zero (standard error 1.38 pp)
- Unprofitable campaigns correctly declined: **28 of 333** (**8.4%** class-0 recall)
- "Always decline everything" would score 25.15% — worse than any targeting strategy

The confidence interval resamples test rows and recomputes **both** the model's success
rate and the baseline's on the same resample, so it accounts for the two moving together
rather than treating the baseline as a fixed constant.

### What this number is, and what it is not

It is an estimate of **improved targeting decisions measured offline**. It is *not*
confirmed ROI, for three reasons:

1. Accuracy weights every campaign equally; euro impact does not. A large campaign and a
   small one count the same here.
2. It assumes the historical label is a faithful proxy for future profitability.
3. It assumes deploying the model does not change *which* campaigns get proposed. It will.

Converting this into a revenue claim requires the A/B test in §5, not arithmetic.

Two further caveats belong next to the headline rather than in a footnote. Nested
cross-validation puts the selection optimism at roughly **1.2 accuracy points**, so the
development figures behind this comparison are optimistic even though the test figure is
not. And the ability to decline unprofitable campaigns — the 8.4% above — is simultaneously
the weakest part of the system and the part with the clearest business value.

### A more useful operating point

At full coverage the model decides every campaign at 54.83%. Applying the confidence gate
selected on the calibration split (threshold 0.60, frozen before test):

| | Coverage | Accuracy on decided campaigns |
|---|---|---|
| Decide everything | 100% | 54.83% |
| Confidence ≥ 0.60 | **34.8%** | **71.6%** |

The deployable proposal is *automate the confident third, escalate the rest.* The A/B test
should measure model-versus-baseline **on the same subset**, since the baseline also scores
above its 46.45% average on easy campaigns — comparing a gated model against an ungated
baseline would overstate the gain.

### Translating into money

With `C` campaigns per year at average spend `S` and average return `R` on a correctly
targeted campaign:

```
Additional value = C x (lift_pp / 100) x R          # better targeting
                 + C x P(class 0) x avoidance_rate x S   # avoided waste
```

State the assumed `C`, `S` and `R` explicitly in the report — the point is the method,
not a precise euro figure.

## 3. Why offline numbers are not enough

The historical data records outcomes for campaigns that *were run*, under the previous
targeting policy. Three caveats belong in the report:

1. **Selection bias** — we only observe the ROI of campaigns the company chose to run.
2. **The counterfactual problem starts at deployment, not before.** Historically the
   company targeted *both* groups each round and measured the ROI of both, so the
   historical record is complete — there is no missing counterfactual in the training
   data. But the model's own policy targets only one group (or none), so **from the day
   it is deployed, the untargeted group's outcome is no longer observed**. The complete
   dataset this model was trained on could not be regenerated under the model's own
   policy — which is precisely why the exploration and propensity-logging scheme in
   §5.1 must ship with the model rather than being added later.
3. **Non-stationarity** — customer behaviour and the marketplace change over time, so a
   model trained on old campaigns degrades.

These make an online test necessary, not optional.

## 4. Offline validation (before any deployment)

1. **Stratified 5-fold cross-validation** on the training split; report mean ± std of
   **accuracy** — accuracy *is* the campaign success rate, so it is the only candidate
   metric that is directly the business objective. Overlapping intervals between models
   mean the difference is not real, and on this dataset they overlap.
2. **Held-out test set *selected on* exactly once, at the end** — and *enforced*, not
   promised. The candidate loop contains no reference to the test set, and
   `test_only_the_champion_is_evaluated_on_the_test_set` fails if one is reintroduced. This
   is stated as a mechanism rather than an intention because an earlier version of this
   project violated it while the report claimed otherwise.

   The wording is deliberate. After the champion is frozen the test set is *read* several
   times to report different quantities; what must happen only once is the **selection**.
   The distinction is the whole content of the guarantee, and blurring it into "touched
   once" makes a claim that a five-second grep disproves.

3. **A stated rule that the holdout is not a virgin test set.** Earlier phases of this
   project cross-validated over the full dataset and ran target-aware exploratory analysis
   on it. Re-splitting under a new `holdout_seed` reshuffles rows the project has already
   collectively seen; **no seed change undoes that**. The honest description is a *reused
   out-of-sample confirmation set*, not an untouched final test set, and the figures it
   produces should be read as confirmation rather than as an unbiased estimate of future
   performance. Genuinely prospective validation requires new campaigns or the A/B test in
   §5 — there is no offline substitute.
3. **Temporal validation** — if the data carries any time ordering, train on older
   campaigns and test on newer ones. This is the honest estimate of future performance.
4. **Leakage audit** — confirm `g1_21`, `g2_21`, `c_28` are absent from the feature
   matrix. Report the inflated score they produce as evidence the trap was seen.
5. **Bootstrap confidence interval** on the lift: resample the test set 1000 times and
   report the 95% interval, so the estimate carries uncertainty.
6. **Sanity checks** — class-balance-aware metrics, per-class recall (especially
   class 0), calibration curve of predicted probabilities.
7. **Three disjoint sets, not two.** Anything *selected* — the calibrator, a decision
   threshold, the confidence gate — is selected on a dedicated calibration split, never on
   test. Fitting a calibrator and choosing whether to keep it on the same rows is selection
   on the fitting set; both are avoided by splitting the calibration data in half again.
8. **A stated minimum improvement for every selection rule.** Calibration is applied only
   if ECE falls by ≥ 10%. Without a margin, a rule accepts any improvement, including one
   indistinguishable from noise — which on a 530-row selection split describes most of them.
   Here isotonic delivered 6.1% and was rejected.

## 5. Online validation (the decisive test)

**Design:** randomised A/B test at the campaign level.

| Item | Choice |
|---|---|
| Unit of randomisation | Campaign |
| Control | Current targeting process |
| Treatment | Model-recommended group; skip when class 0 is predicted |
| Primary metric | ROI per campaign |
| Secondary metrics | Success rate, revenue per campaign, budget spent |
| Guardrails | Total revenue, campaigns skipped, per-segment fairness |
| Allocation | 50/50, or a staged 10% → 50% ramp |

**Sample size.** For a two-sided test at α = 0.05 with 80% power, detecting a difference
in success-rate proportions `p1` vs `p2`:

```
n per arm = 2 x (z_(1-α/2) + z_(1-β))^2 x p_bar x (1 - p_bar) / (p1 - p2)^2
```

Plug in the offline baseline and the observed lift to get the required number of
campaigns per arm, then convert to a duration using the historical campaign cadence.
If the required duration is impractical, say so and propose a staged rollout with
sequential monitoring instead of pretending the test is feasible.

**Stopping rules.** Fix the horizon in advance. If interim looks are needed, use a
sequential method (e.g. alpha spending) rather than peeking.

## 5.1 Learning from logged decisions — inverse propensity scoring

Once deployed, the model determines which campaigns run, so the data it generates is
**not** a random sample. Two distinct biases appear:

1. **Censoring.** A "do not run" recommendation means the outcome is never observed. Those
   campaigns vanish from future training data entirely.
2. **Confounding.** Among campaigns that *do* run, the ones the model liked are
   over-represented. Training naively on this data teaches the next model to agree with the
   current one rather than with reality.

This is a partial-feedback (contextual bandit) problem, not a supervised one, and it cannot
be repaired retrospectively — the logging has to be designed for it before launch.

### What to log per decision

| Field | Purpose |
|---|---|
| `decision_id`, `timestamp` | Join key to the realised outcome |
| `features` | The exact 67 inputs used |
| `predicted_probabilities` | Calibrated, all three classes |
| `action_taken` | What was actually done — may differ from the recommendation |
| `propensity` | **P(this action was taken)** under the deployed policy |
| `exploration` | Whether this was a forced exploration sample |
| `model_version`, `dataset_sha256` | Lineage |

`propensity` is the field that makes everything else usable. Under the current policy it is
`1 - exploration_rate` for the recommended action, and `exploration_rate / 2` for each
action reachable through exploration.

### How the next model uses it

Weight each logged campaign by the inverse of its propensity, so under-sampled actions
count proportionally more:

```
V(new policy) = (1/n) * sum_i [ reward_i * 1{new_policy(x_i) == action_i} / propensity_i ]
```

This yields an **unbiased offline estimate of a candidate policy's value from logged data
alone** — a new model can be evaluated against real campaigns it never chose. Clip
propensity weights (typically at 10-20) or use the self-normalised estimator to bound
variance, since small propensities otherwise dominate the estimate.

### Practical consequences

* **`exploration_rate` cannot be zero *if IPS is wanted*.** With no exploration,
  propensities for unchosen actions are zero, the estimator divides by zero, and no offline
  policy evaluation is possible at all.
* **The shipped default is nevertheless `0.0`, deliberately.** A non-zero exploration rate
  spends real budget on deliberately sub-optimal campaigns. That is a sound investment and
  it is *not a library default's decision to make* — a config file should not quietly commit
  a marketing team to misallocating 5% of its spend. The value is a required, explicit
  input before the logging scheme is switched on.
* **The trade-off, stated so it can be decided rather than inherited:** at 5% exploration,
  roughly 1 campaign in 20 is targeted sub-optimally on purpose. In exchange, every future
  candidate model can be evaluated against real campaigns it never chose, without running a
  new A/B test each time. Over a year that is usually much cheaper than the alternative —
  but the number belongs to the business, not to `config.py`.
* **Exploration samples are the only unbiased subset.** Report performance on them
  separately — that is the honest estimate.

## 6. Post-deployment monitoring

| Signal | Method | Action |
|---|---|---|
| Feature drift | PSI / KS test on incoming features vs. training | Investigate at PSI > 0.2 |
| Prediction drift | Predicted class distribution vs. training | Alert on sustained shift |
| Performance decay | Realised success rate vs. the 46.45% naive baseline, once outcomes land | Retire the model if it falls below the baseline — at that point it subtracts value |
| **Position-convention drift** | Distribution of `group_1` vs `group_2` inputs; symmetry violation rate on recent traffic | **Alert loudly.** 53.8% of predictions flip under a group swap, so a change in how upstream assigns "group 1" degrades the model sharply and silently |
| Coverage under the gate | Share of requests above the 0.60 confidence threshold | Investigate a sustained fall — it means the model is losing confidence before it loses accuracy |
| Data quality | Null rate, out-of-range values per feature | Block and alert |
| Service health | Latency p50/p95, error rate, uptime | Standard SRE alerting |

**Retraining cadence:** scheduled monthly, plus event-driven on a drift alert. Every
candidate must beat the deployed champion on CV **accuracy** *and* business lift, **by more
than one fold standard deviation** — on this data a 0.4-point margin is not a win. Artifacts
are versioned and every run is recorded in MLflow with its parameters, metrics and dataset
hash, so rollback is a config change and the lineage of any deployed model is recoverable.

## 7. Honest limitations

- The lift estimate assumes the historical target is a faithful proxy for future ROI, and
  is an offline estimate of decision quality rather than confirmed revenue.
- Class 0 has 333 test rows and 8.4% recall — the widest uncertainty of any figure here,
  and the class with the clearest business value.
- **Independence between campaigns cannot be verified.** No campaign or customer identifier
  exists in the data. If the same group recurs across rows, the effective sample size is
  below 6,620 and every interval quoted here is too narrow.
- **No temporal validation is possible.** No time-ordering column was detected, so
  generalisation to future campaigns — as opposed to held-out ones — is untested. This is
  the single cheapest thing the data owner could fix.
- Feature meanings are anonymised, so causal interpretation of importances is not possible
  — the model supports the decision, it does not explain the market.
- The champion's margin over the runner-up is inside noise, and nested CV ranks them in the
  opposite order. The defensible claim is that selection happened before the test set was
  opened, not that the best model was found.
