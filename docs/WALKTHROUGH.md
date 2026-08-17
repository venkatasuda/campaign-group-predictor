# Walkthrough — one campaign, end to end

*Read this first. It traces a single campaign through every component, in order, with the
file and function that handles each step. Ten minutes here replaces an hour of reading
`src/`.*

There are two journeys. **Part 1** is offline: one row of the CSV becoming a trained
model. **Part 2** is online: one HTTP request becoming a targeting recommendation. They
meet at `artifacts/model.pkl`.

---

## Part 1 — from a CSV row to a model artifact

### The starting point

`data/customerGroups.csv` holds 6,620 past campaigns, one per row, 71 columns:

| Columns | Count | Meaning |
|---|---|---|
| `g1_1 … g1_20` | 20 | Group 1 characteristics, known **before** the campaign |
| `g2_1 … g2_20` | 20 | Group 2 characteristics, known **before** |
| `c_1 … c_27` | 27 | Comparison variables, known **before** |
| `g1_21`, `g2_21`, `c_28` | 3 | Recorded **after** the campaign ran |
| `target` | 1 | 0 = neither profitable, 1 = group 1, 2 = group 2 |

### Step 1 — the contract is declared once

**`src/constants.py`**

`BASE_FEATURES` is the 67 columns available at prediction time. `LEAKAGE_FEATURES` is the
three that are not. Every other module imports from here, so training and serving cannot
drift apart — if a column is renamed, one file changes.

### Step 2 — the three post-campaign columns are removed

**`src/training/pipeline.py` → `split_features_target()`**

Returns the 67 features and the target, dropping `g1_21`, `g2_21`, `c_28`.

**Why they are dropped is not what you'd expect.** Measured, they inflate cross-validated
macro F1 by only **+0.0038** — they barely leak at all. They are excluded because they *do
not exist* when a campaign is being planned. Availability, not correlation, is the
criterion. See `reports/REPORT.md` §3.

### Step 3 — the test set is created and then left alone

**`src/training/splits.py` → `make_split()`**

A stratified 80/20 split. `detect_time_ordering()` runs first and finds no column behaving
like a timestamp, so a temporal split is not available — that limitation is recorded rather
than glossed over.

The training portion is split again to produce a **calibration set**. From this point until
the final evaluation, nothing reads the test set. That rule is what makes the reported
number mean anything.

### Step 4 — features are derived inside the pipeline, not before it

**`src/features.py` → `PairwiseFeatureBuilder`**

Adds `diff_i = g1_i - g2_i` and `ratio_i` for each paired variable, because the label is a
comparison and a tree needs many splits to approximate a subtraction.

It lives **inside** the sklearn `Pipeline` (`src/training/pipeline.py → build_pipeline()`),
alongside the imputer and scaler. That placement is deliberate: fitted inside each CV fold,
it cannot leak statistics from validation rows into training ones. It also means the exact
same transformation runs at serving time, because the whole pipeline is what gets
serialised.

> Measured honestly, these derived features add nothing on this dataset (0.5699 with them,
> 0.5702 without). Reported in `reports/advanced_experiments.json` rather than quietly
> retained.

### Step 5 — training

**`src/training/train.py` → `train()`**

One long linear function, deliberately, with eight labelled phases in its docstring. A
training run *is* a sequence performed once in order, and splitting it across helpers would
hide the property that matters most: the test set is created in phase 2 and not touched
again until phase 6.

The phases: load → split → symmetry preconditions → model zoo → champion selection → test
evaluation → calibration and explainability → persist.

**Phase 3 is the interesting one.** Before mirroring rows to double the training data, the
code tests whether the two group positions are actually interchangeable. They are not —
7 of 20 paired variables differ significantly — so augmentation is *refused*, not warned
about. `src/symmetry.py → validate_group_exchangeability()`.

**Phase 5 selects on cross-validation, never on the test score.** The candidate loop contains
no reference to `x_test` at all, and `test_only_the_champion_is_evaluated_on_the_test_set`
fails if one is reintroduced — which is how the property was lost the first time. An override
via `--champion` remains possible but is recorded in `metrics.json` as
`champion_overridden`, so a reader can always see whether a judgement was made. **It was not:
the shipped model is the automatic winner on cross-validated accuracy.**

### Step 6 — the artifact

`artifacts/model.pkl` is a joblib dict, not a bare pipeline, so metadata travels with the
weights: `model_name`, `model_version`, `trained_at`, offline metrics, and the dataset's
`dataset_sha256` for lineage. `artifacts/metrics.json` holds every model's scores and every
diagnostic from the run.

---

## Part 2 — from an HTTP request to a recommendation

### Step 7 — the model is loaded once

**`src/registry.py` → `ModelRegistry`** *(Singleton)*

Loaded during FastAPI's `lifespan` startup (`src/api/main.py`), then shared across every
request. Deserialising a pipeline per request would dominate the latency budget.

If the artifact is missing, `allow_baseline_fallback` decides whether to degrade to the
majority-class baseline or fail the deploy. **In production it is `false`** — a service
that starts healthy and silently serves a baseline is the worst available failure mode.

### Step 8 — the request

```http
POST /predict
Content-Type: application/json

{
  "group_1":    {"g1_1": 0.42, "g1_2": 1.7,  ...},   // all 20
  "group_2":    {"g2_1": 0.31, "g2_2": 2.1,  ...},   // all 20
  "comparison": {"c_1": 0.08,  "c_2": -0.44, ...}    // all 27
}
```

### Step 9 — validation, before the model is touched

**`src/schemas.py` → `ComparisonRequest`**

Four checks, each with its own error message:

| Check | Response |
|---|---|
| Contains `g1_21`, `g2_21` or `c_28` | **422** — "must not contain post-campaign variables" |
| Missing a required key | **422** — names the missing keys |
| Unexpected key inside a block | **422** — names them |
| Unexpected key at the top level | **422** — `extra="forbid"` |

This is the **third independent place** the leakage rule is enforced, after the training
split and the feature adapter. A unit test fails if any of the three lets one through.

### Step 10 — JSON becomes a DataFrame

**`src/features.py` → `FeatureTransformer.from_payload()`** *(Adapter)*

Produces a 1 × 67 frame in the canonical column order the pipeline expects. The model
never sees HTTP concepts; the API never sees column ordering.

### Step 11 — prediction

**`src/predictors.py` → `SklearnPipelinePredictor.predict()`** *(Strategy)*

The frame goes through the serialised pipeline — pairwise features, imputation, scaling,
model — and comes back as a class plus probabilities. The API depends only on
`BasePredictor`, so swapping the model requires no API change.

### Step 12 — the decision, which is not `argmax`

**`src/decision.py` → `DecisionPolicy.decide()`**

`argmax` answers *"which outcome is most likely?"*. The business asks *"which action
maximises expected return?"*. Those differ whenever mistakes cost different amounts —
targeting the wrong group burns the budget, declining a profitable campaign forgoes margin,
and the two are not equal.

The policy computes the expected cost of each action under an explicit cost matrix and
picks the minimum. When the top two actions are nearly tied it sets `review_required`, and
the campaign routes to a human instead.

> The cost figures are **assumptions on a relative scale, not measured euros.** They are
> presented as a sensitivity analysis across three ratios. Confirming the real numbers with
> the marketing team is the cheapest available improvement to this system.

### Step 13 — the response

```json
{
  "predicted_class": 2,
  "label": "group_2",
  "recommended_action": "Target customer group 2.",
  "confidence": 0.61,
  "probabilities": {"no_group_profitable": 0.14, "group_1": 0.25, "group_2": 0.61},
  "decision": {
    "action": 2,
    "recommended_action": "Target customer group 2.",
    "expected_costs": {"do_not_run": 0.39, "target_group_1": 0.50, "target_group_2": -0.22},
    "differs_from_argmax": false,
    "review_required": false,
    "rationale": "Lowest expected cost, clear of the review margin."
  }
}
```

Every response carries `X-Request-ID`, and the same ID appears in the log line for that
request — so a recommendation questioned six weeks later can be traced to the model version
and probabilities that produced it.

---

## The three things this design protects

**1. A post-campaign column cannot reach the model.** Blocked in the training split, the
feature adapter and the HTTP schema, with a test guarding each.

**2. The test set decides nothing.** Model family, feature rankings, the automation
threshold, the decline rule and the calibration method are all selected on cross-validation
or a held-out calibration split, then frozen. The test set is measured once and used only
to report. Two of these were originally selected on test and moved after review — see
`reports/REPORT.md` limitations.

**3. The model and the decision are separate objects.** Changing what a mistake costs is a
config change, not a retrain.

---

## Where to look next

| Question | File |
|---|---|
| What did the data actually say? | `notebooks/01_analysis.ipynb` |
| What are the numbers and the caveats? | `reports/REPORT.md` |
| How do the components fit together? | `docs/architecture.md` |
| What is this model *not* fit for? | `docs/model_card.md` |
| How would we prove the lift is real? | `docs/validation_plan.md` |
| Does it work? | `pytest` — or the live `/health` endpoint |
