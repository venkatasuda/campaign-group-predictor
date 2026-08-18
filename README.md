# Predicting Profitable Customer Groups — End-to-End ML Solution

> **Confidential.** The task specification and the accompanying dataset are marked
> *LP-Internal* and state that further distribution and reproduction are prohibited.
> This repository is therefore **private**: the dataset is never committed, the task
> document is not reproduced here, and no data source link is included. Share it as a
> private repository or a zip archive only.

The company in this exercise is an **online marketplace**. Over many years it ran
marketing campaigns, each round targeting two different customer groups, and evaluated
the ROI of both groups afterwards to learn which was the more profitable to target.

Given the pre-campaign characteristics of two customer groups, this service predicts
which group the next campaign should target — or whether neither group is worth
targeting at all.

| Target class | Meaning | Action |
|---|---|---|
| `0` | Neither group was profitable | Do not run the campaign |
| `1` | Group 1 was most profitable | Target group 1 |
| `2` | Group 2 was most profitable | Target group 2 |

---

## Deliverables map

> **New here? Read [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) first.** It traces one
> campaign from a CSV row to a JSON recommendation, naming the file that handles each step.
> Ten minutes there replaces an hour of reading `src/`.

| Challenge item | Where |
|---|---|
| ML Q1 — outcome percentages | `notebooks/01_analysis.ipynb`, `artifacts/metrics.json`, `src/training/evaluation.py::campaign_outcome_distribution` |
| ML Q2 — predictive model | `src/training/`, `src/predictors.py` |
| ML Q3 — lift + validation | `docs/validation_plan.md`, `src/training/evaluation.py::estimate_business_lift` |
| Eng 1 — architecture + sequence diagrams | `docs/architecture.md` |
| Eng 2 — deployed API, OOP, unit tests | `src/api/`, `tests/`, `Dockerfile` |
| Infrastructure as code | `terraform/` |
| Model card + ADR | `docs/model_card.md` |
| Report | `reports/` |

## Headline result

Champion **random forest**, selected automatically on cross-validated accuracy and scored
**once** on a 1,324-campaign test set it never saw during development.

| | |
|---|---|
| Campaign success rate | **54.83%** |
| Best naive strategy ("always group 1") | 46.45% |
| **Lift** | **+8.38 pp**, 95% CI [+5.66, +11.18] |
| With a frozen confidence gate | **71.6%** accuracy on the **34.8%** of campaigns the model is confident about |

Three things this number is not: it is not confirmed ROI (accuracy weights every campaign
equally, euros do not); it is not a claim that random forest is the best model (its 0.40-point
margin over XGBoost is inside noise, and nested CV ranks them the other way); and it is not
evenly distributed across classes — **class-0 recall is 8.4%**, and class 0 is where money is
saved rather than earned. All three are treated as findings rather than footnotes in
[`reports/REPORT.md`](reports/REPORT.md).

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-optional.txt            # optional — widens the model zoo

# 2. Data — place the provided customerGroups.csv into data/
#    (not committed: the dataset is confidential and must not be redistributed)

# 3. Train — use `make train`, or the full command below. The flags are not optional:
#    --holdout-seed fixes WHICH campaigns are held out (omitting it falls back to 42 and
#    produces a different test set), and --calibrate reserves the 1,060-row calibration
#    split that makes thresholds selectable without touching test. Every number in
#    reports/ and docs/ comes from this exact invocation.
#
#    Verify with `make verify-artifact` before deploying: it asserts the artifact on disk
#    is the documented run, because a successful training run only proves *a* model was
#    written, not that it is the right one.
python -m src.training.train \
    --data data/customerGroups.csv \
    --out artifacts \
    --calibrate \
    --holdout-seed 20260819

# Expected: champion random_forest, split 4,236 / 1,060 / 1,324, test accuracy 0.5483,
# lift +8.38 pp, calibration evaluated and REJECTED.
python -c "import json; m=json.load(open('artifacts/metrics.json')); c=m['champion_model']; print(m['holdout_seed'], c, m['models'][c]['accuracy'])"

# 4. Serve
uvicorn src.api.main:app --reload --port 8000
#    Interactive docs: http://localhost:8000/docs

# 5. Frontend (separate terminal)
streamlit run frontend/app.py

# 6. Tests
pytest
```

`make install | train | test | lint | api | frontend | docker-build | deploy` wrap all of the above.

### Experiment tracking

```bash
pip install -e ".[tracking]"
python -m src.training.train --data data/customerGroups.csv --out artifacts
mlflow ui                      # http://localhost:5000
```

Runs are written to a local SQLite store (`mlflow.db`) rather than the plain `mlruns/` file
store, for two reasons: it is what `mlflow ui` reads by default — writing to one and
reading from the other produces an empty dashboard and no error — and the **Model
Registry** requires a database-backed store, so versioning and stage promotion are
available without migrating later.

Every training run logs its parameters, one nested child run per candidate model with the
full metric set, the champion's metrics at the top level, and the **dataset SHA-256** as a
tag — so a served model traces back to the run and the exact data that produced it.

`artifacts/metrics.json` is still written: it travels with the model artifact, is readable
without MLflow installed, and diffs cleanly in version control. MLflow adds history *across*
runs, which a single overwritten file cannot.

> This replaced a file-per-run approach. It worked until the same job was run three times
> with different search strategies, at which point output directories started accumulating
> (`artifacts_tuned/`, `artifacts_verify/`) — a naming convention standing in for run
> history. That is the point at which experiment tracking stops being optional.

Tracking is an **optional dependency** and degrades to a no-op when absent. A training run
must not fail because of the layer that observes it.

### Quality gates

```bash
pip install pre-commit && pre-commit install
```

Runs ruff, black and file hygiene checks before each commit — plus one hook that matters
more than the rest: **it refuses any commit that stages the dataset**.

`.gitignore` already excludes `data/*.csv`, but `.gitignore` is advisory — `git add -f`
overrides it silently. Once a confidential file reaches history, removing it means
rewriting every commit after it and treating the contents as disclosed regardless, because
anyone who cloned in between still has it. The same check runs in CI, so it holds for
anyone who did not install the hooks.

**What this check does not do, stated precisely.** It inspects `git diff --cached` — the
**staged** changes of the commit being made. It therefore does *not* scan the working tree,
does *not* audit existing history, and does *not* inspect a release archive. Running it in a
repository where `data/customerGroups.csv` merely *exists* will pass, correctly, because
nothing is staged.

That distinction matters because relying on it for more than it does is exactly how the
dataset reached a submission archive: the commit guard held, and the packaging step had no
guard at all. Archive safety is a separate mechanism — `scripts/build_release.py` builds from
an **allowlist** and refuses to write if any `.csv` outside three named derived tables appears
in the manifest.

### Installing as a package

The project is an installable distribution, not just a folder of scripts:

```bash
pip install -e .                    # runtime only
pip install -e ".[models]"          # + CatBoost, LightGBM, XGBoost, SHAP
pip install -e ".[all]"             # + notebook, frontend, test and lint tooling
```

That installs a console entry point, so training is invoked by name rather than by
remembering a module path — the same interface a developer uses and a scheduled job would
call:

```bash
campaign-train --data data/customerGroups.csv --out artifacts \
    --calibrate --holdout-seed 20260819
```

**The two flags are not optional.** `--holdout-seed` fixes *which* campaigns are held out —
omitting it falls back to 42 and produces a different test set, and therefore different
accuracy, lift, class-0 recall and symmetry figures. `--calibrate` reserves the 1,060-row
calibration split that makes thresholds and the calibrator selectable without touching test;
without it the split is 5,296 / – / 1,324 and every reported number shifts.

This is written out because a bare `--data ... --out artifacts` once overwrote the canonical
artifact with a differently-split experiment, and the deployed service then served a model
whose accuracy matched no document describing it. `make train` carries the flags; verify with
`make verify-artifact`, which asserts the artifact on disk is the documented run and exits
non-zero otherwise.

### Dependency files

Runtime dependencies live in `pyproject.toml`; the requirements files specify three
*environments*. Both exist on purpose — the first is the contract of the library (what a
consumer needs to import it), the second describes what to install for a given job. Someone
installing the package should not receive JupyterLab; a contributor should.

| File | Used by | Contents |
|---|---|---|
| `pyproject.toml` | `pip install .` | Runtime dependencies and optional extras |
| `requirements.txt` | Development | Notebook, frontend, tests, linters |
| `requirements-serve.txt` | **The API container** | Only what the service imports |
| `requirements-optional.txt` | Full model zoo + tracking | CatBoost, LightGBM, XGBoost, SHAP, MLflow — all guarded imports |

The API image installs **only** `requirements-serve.txt`. JupyterLab, Streamlit, matplotlib
and pytest have no role in answering a prediction, and shipping them costs image size,
Cloud Run cold-start latency and CVE surface for code the process never loads.

**Pinning is selective, not uniform.** The question worth asking of each dependency is *what
must this be identical to?* — and only the four that deserialise the pickle have to be
identical to anything:

| Packages | Constraint | Why |
|---|---|---|
| `scikit-learn`, `numpy`, `scipy`, `joblib` | **`==` exact** | These unpickle the model. A minor bump can change an estimator's internal layout, and the failure mode is the worst available: the image builds, the container starts, `/health` passes, and the **first prediction** fails |
| `pandas` | `~=` compatible | Used by the feature adapter, not by the pickle |
| `fastapi`, `uvicorn`, `pydantic` | `>=,<` ranges | The HTTP layer never touches the artifact. Pinning it exactly buys no safety and declines security patches in the component most exposed to the internet |

The container also runs **Python 3.12** to match the interpreter that produced
`artifacts/model.pkl`. Serving on 3.11 while training on 3.12 worked, but by luck rather than
design — a pickle carries references to the classes that created it.

`metrics.json` records these versions in an `environment` block. That duplicates the artifact
deliberately: reading them from the artifact requires unpickling it, which needs the very
libraries you are trying to identify — circular exactly when it matters.

> **If you retrain and the champion changes family**, a pickled CatBoost/LightGBM/XGBoost
> pipeline cannot be unpickled without its library. Add the matching line to
> `requirements-serve.txt` and rebuild — otherwise the container starts healthy and fails
> on the first request.

### After deploying

```bash
make deploy
make verify-deploy      # asserts /health reports model_loaded: true
```

Cloud Run reports a deploy as successful once the container answers its health probe —
which it does even if the model artifact never reached the image. `verify-deploy` exists
to catch a green deploy that is serving no model, and exits non-zero so it can gate a
release step.

---

## Reading guide — where to start

The codebase is ~1,400 statements. If you have fifteen minutes, these four files carry the
substance of the solution:

| Read | Why |
|---|---|
| **`src/constants.py`** | The dataset contract in one screen: which 67 columns are usable, which 3 are excluded and why. Everything else derives from this. |
| **`src/symmetry.py`** | The most interesting analysis. Tests whether the two group positions are interchangeable — they are not — and therefore refuses the augmentation that was originally planned. |
| **`src/decision.py`** | Where a prediction becomes a business action: expected-cost minimisation over an explicit cost matrix, with abstention and exploration. |
| **`src/training/train.py`** | The orchestration, and the record of every methodological choice: the three-way split protocol, the selection metric, the calibration rule and its minimum-improvement guard, and the phase boundary after which the test set may be touched. |
| **`src/training/diagnostics.py`** | What turns a 57%-accurate classifier into a deployable policy: learning curve, confidence operating points, feature ablation, and the separation of model from decision rule. |

Then `src/api/` for the serving layer and `tests/` for the contracts each module guarantees.

**Documentation convention.** Module docstrings explain *why a module exists*; function
docstrings give the contract; inline comments explain *decisions that the code cannot
convey on its own* — usually a non-obvious constraint, a library quirk, or a rejected
alternative. Mechanical narration of statements is deliberately absent.

## Project structure

```
campaign-group-predictor/
├── src/
│   ├── constants.py          # dataset contract: 67 features + 3 leakage columns
│   ├── config.py             # env-driven settings (12-factor)
│   ├── exceptions.py         # domain exception hierarchy
│   ├── logging_config.py     # structured logging
│   ├── schemas.py            # Pydantic request/response contract
│   ├── features.py           # PairwiseFeatureBuilder + FeatureTransformer (Adapter)
│   ├── symmetry.py           # group-swap augmentation + invariance measurement
│   ├── calibration.py        # isotonic/sigmoid calibration, ECE, reliability curves
│   ├── decision.py           # cost matrix, expected-cost rule, abstention, exploration
│   ├── drift.py              # PSI reference capture + drift detection
│   ├── explainability.py     # permutation importance + SHAP
│   ├── predictors.py         # BasePredictor (Strategy) + implementations
│   ├── factory.py            # ModelFactory (Factory + registry)
│   ├── registry.py           # ModelRegistry (Singleton)
│   ├── api/
│   │   ├── main.py           # create_app() (Application Factory), lifespan, handlers
│   │   ├── routes.py         # /predict, /predict/batch, /health, /model/info
│   │   └── dependencies.py   # dependency injection providers
│   └── training/
│       ├── pipeline.py       # dataset loading, leakage removal, pipeline assembly
│       ├── splits.py         # temporal-ordering detection, split strategies, redundancy
│       ├── tuning.py         # randomised hyperparameter search spaces
│       ├── evaluation.py     # metrics + business lift (ML Q1 and Q3)
│       ├── diagnostics.py    # learning curve, operating points, ablation, ensembling
│       └── train.py          # training CLI
├── frontend/app.py           # Streamlit UI - a thin client, no business logic
├── tests/                    # 427 tests across 17 modules
├── docs/                     # architecture.md, validation_plan.md, model_card.md, WALKTHROUGH.md
├── notebooks/
│   ├── 01_analysis.ipynb     # the analysis narrative, reproducible end to end
│   └── analysis_support.py   # presentation helpers - NOT in src/, see below
├── reports/                  # REPORT.md, findings.json, figures/, PEER_REVIEW_SUMMARY.md
├── artifacts/                # model.pkl, metrics.json (git-ignored)
├── data/                     # customerGroups.csv (git-ignored, confidential)
├── Dockerfile / Dockerfile.frontend / cloudbuild.frontend.yaml
├── terraform/                # Cloud Run, IAM, Artifact Registry
└── .github/workflows/        # ci.yml (push/PR) + deploy.yml (manual)
```

**Why `analysis_support.py` sits in `notebooks/` and not `src/`.** It holds the notebook's
presentation helpers — table formatting, chart assembly — and imports matplotlib, seaborn and
`IPython.display`. Three consequences follow, in order of how much they matter:

1. `src/` is the **deployed** package. `requirements-serve.txt` ships scikit-learn, pandas and
   FastAPI and nothing else. If these helpers lived in `src/`, the API could import a
   dependency the container does not have — a failure that appears only at runtime, in
   production.
2. Coverage is measured on `src/` against a `fail_under` floor. Adding several hundred lines
   of chart formatting would either break the gate or force tests asserting the colour of a
   heatmap.
3. `.gcloudignore` excludes `notebooks/`, so none of it reaches the serving image.

The boundary is *product code versus analysis code*, not *importable Python versus scripts*.

### Decision-support diagnostics

`src/training/diagnostics.py` answers the questions a stakeholder asks *after* the headline
accuracy, each of which changes what to do next:

| Function | Question it answers |
|---|---|
| `learning_curve_report` | Would collecting more campaigns help, or is the model feature-limited? |
| `operating_points` | What does requiring a minimum confidence buy in accuracy, and cost in coverage? |
| `select_confidence_threshold` | Picks that threshold on a **validation** split, so it can be frozen before the test set is touched |
| `abstention_rule_comparison` | Separates the *model* from the *decision rule* — can a `P(class 0) ≥ τ` rule recover unprofitable campaigns that `argmax` never surfaces? |
| `feature_block_ablation` | Which feature blocks earn their place? |
| `feature_count_sweep` | How few features are actually needed? |
| `ensemble_experiment` | Does voting or stacking beat the best single model, or is the diversity nominal? |

Two of these exist specifically to keep the test set clean:
`select_confidence_threshold` prevents the automation threshold from being chosen by
inspecting test results, and the feature ranking passed to the sweep and ablation is
derived from the calibration split rather than from test.

---

## The leakage trap

The task states that `g1_21`, `g2_21` and `c_28` were recorded **after** the campaign ran.
They correlate strongly with the target but do not exist when the model is called in
production. Training on them yields a near-perfect offline score and a worthless model.

They are removed in three independent places:

1. `split_features_target()` drops them before training.
2. `FeatureTransformer` rejects any request containing them.
3. The Pydantic schemas reject them at the HTTP boundary with a 422.

Only the **67** pre-campaign variables are used: `g1_1..g1_20`, `g2_1..g2_20`, `c_1..c_27`.

---

## Modelling approach

1. **Pairwise features** — the task is a comparison, so `diff_i = g1_i - g2_i` and
   `ratio_i = g1_i / |g2_i|` are added for all 20 paired variables. A tree model would
   otherwise need many splits to approximate a single subtraction.
2. **Group-swap symmetry** — which group is called "1" is arbitrary, so the problem is
   exactly antisymmetric: swapping the groups must swap the prediction 1↔2 and leave 0
   alone. `src/symmetry.py` **enforces** it (mirrored training rows, via
   `--augment-symmetry`), **verifies** it (a KS test per comparison column catches
   ratio-type features that sign-flipping would corrupt), and **measures** it
   (`violation_rate` reported for every model). A model with a high violation rate is
   keying on position, not on signal.

   Augmentation forces grouped cross-validation: a campaign and its mirror are not
   independent, so `augment_with_swapped_grouped` returns group ids and training uses
   `StratifiedGroupKFold`. Under plain `StratifiedKFold` the mirror of a validation row
   appears in the training folds and every CV score is inflated.
3. **One pipeline object** — pairwise features → median imputation → standardisation →
   classifier. The same fitted object is serialised and served, so training and serving
   transformations cannot drift apart.
4. **Model zoo spanning four families** — linear (logistic regression), bagged trees
   (random forest), boosted trees (CatBoost, LightGBM, XGBoost, HistGradientBoosting),
   and a neural baseline (MLP). All seven are compared under identical folds. **Random
   forest won on cross-validated accuracy (0.5781 ± 0.0051)** and was promoted
   automatically — no human override, and the winner is not re-checked against the test
   score.
5. **The test set is *selected on* once, and that is enforced rather than promised.** The
   candidate loop contains no reference to `x_test`; only the locked champion is scored.
   `test_only_the_champion_is_evaluated_on_the_test_set` fails if that changes. An earlier
   version of this repository scored every candidate on test "for the leaderboard" while
   selecting on CV — selection bias needs only visibility, not intent, so the property is
   now a test rather than a sentence in a report.

   **The precise claim, because the loose one is false.** After the champion is locked, the
   test set is *read* several times — metrics, symmetry invariance, calibration reporting,
   decision-policy evaluation, drift. Grep `x_test` in `train.py` and you will find about
   eight hits. That is not contamination: reading a frozen model's predictions to compute a
   different quantity changes nothing, because **no choice depends on the result**. The
   property that matters is *selected on once*. Saying "read once" would be a stronger
   claim and an untrue one, and a reviewer who checks would rightly stop trusting the rest.
6. **Diagnostics before modelling** — `src/training/splits.py` checks whether any column
   encodes a time ordering (if so, use `--split temporal`, because a random split would
   leak the future) and whether the `c_` features are deterministic functions of the
   group blocks.
7. **Metrics** — the selection metric is **accuracy**, because accuracy *is* the campaign
   success rate and so is directly the business objective; macro F1 weights three classes
   equally, which is a different goal and can prefer a model that underperforms the naive
   baseline. Balanced accuracy, macro F1, the confusion matrix and per-class recall are all
   reported alongside it, because accuracy alone would hide that class-0 recall is 8.4%.
8. **Explainability** — permutation importance, computed on the calibration split rather
   than on test, so the feature ranking is not itself a test-set read. A third person can
   see *why* a group is recommended.
9. **Calibration** — probabilities feed a euro-denominated decision rule, so they must mean
   what they say. `--calibrate` fits isotonic (or sigmoid) regression on one half of the
   calibration split, evaluates it on the other half, and **keeps it only if ECE falls by at
   least 10%**. Three disjoint sets, because fitting a calibrator and deciding whether to
   keep it on the same rows is selection on the fitting set. On this data isotonic delivered
   6.1% and was **rejected** — the rule is real, not decorative.
10. **Cost-sensitive decisions** — `argmax` answers "what is most likely?"; the business
    asks "what action maximises expected return?". `src/decision.py` picks the action with
    the lowest expected cost under an explicit cost matrix, and flags near-ties (7.55% of
    test campaigns) for human review instead of automating a coin flip.
11. **Experiment tracking** — every training run is recorded in MLflow with its parameters,
    metrics, artifacts, dataset hash and timestamp, one nested run per candidate. `mlflow ui`
    against the default SQLite backend shows the full history.

### The decision layer

Targeting the wrong group, running a doomed campaign, and skipping a profitable one do
**not** cost the same. The API therefore returns both the prediction and the action:

```json
{
  "predicted_class": 1,
  "confidence": 0.36,
  "probabilities": {"no_group_profitable": 0.34, "group_1": 0.36, "group_2": 0.30},
  "decision": {
    "action": 0,
    "action_label": "do_not_run",
    "recommended_action": "Do not run this campaign - neither group is expected to be profitable.",
    "expected_costs": {"do_not_run": 0.0, "target_group_1": 60.4, "target_group_2": 67.0},
    "differs_from_argmax": true,
    "review_required": false,
    "rationale": "The most likely outcome is 'group_1', but 'do_not_run' has the lower expected cost once the asymmetric cost of each mistake is taken into account."
  }
}
```

Cost inputs live in `.env` (`CAMPAIGN_SPEND`, `PROFIT_IF_CORRECT`, `OPPORTUNITY_WEIGHT`,
`DECISION_REVIEW_MARGIN`). They are **business inputs, not modelling constants** — the
committed values are placeholders on a relative scale. Set `ENABLE_DECISION_LAYER=false`
to return plain predictions.

### Exploration — why declining is a trap

A "do not run" recommendation means the campaign's outcome is **never observed**. The
deployed model therefore censors the data that will train its successor, and the training
set drifts toward cases the current model already likes. That is a partial-feedback
problem, not a supervised one.

`EXPLORATION_RATE` overrides a fraction of decline recommendations so those outcomes keep
arriving. The cost is bounded and known up front — `rate × P(class 0) × campaign spend` —
and every such decision is marked `"exploration": true` so it can be excluded from
performance reporting and used as the clean evaluation sample.

**The default is `0.0`, deliberately.** Exploration spends real budget on deliberately
sub-optimal campaigns; that can be an excellent investment, but it is not a config default's
decision to make. Shipping `0.05` would quietly commit a marketing team to misallocating one
campaign in twenty without anyone having agreed to it. The trade-off is documented in
`docs/validation_plan.md` §5.1 so it can be *decided*, and the value must be set explicitly
before inverse-propensity evaluation is possible at all — with no exploration, propensities
for unchosen actions are zero and the estimator is undefined.

### Drift monitoring

`src/drift.py` captures the training feature distributions (quantile bins, means, missing
rates) into the model artifact at fit time, so PSI can actually be computed later against
a real reference. Conventional thresholds: `< 0.10` no action, `0.10–0.25` investigate,
`≥ 0.25` retrain. Training also runs a sanity check against the hold-out split, which
should report no drift — if it does, something is wrong with the split.

### Training CLI

```bash
# Fast smoke run, one model, no SHAP
python -m src.training.train --data data/customerGroups.csv --fast \
    --models logistic_regression --skip-explain

# Full run: whole zoo, tuned, symmetry-augmented, calibrated
python -m src.training.train --data data/customerGroups.csv --out artifacts \
    --tune --n-iter 25 --augment-symmetry --calibrate

# With real business costs supplied by the marketing team
python -m src.training.train --data data/customerGroups.csv --calibrate \
    --campaign-spend 500 --profit-if-correct 2000 --opportunity-weight 0.5

# Honest protocol if a time ordering was detected
python -m src.training.train --data data/customerGroups.csv --split temporal

# Leakage demonstration for the report — never deploy this model
python -m src.training.train --data data/customerGroups.csv --keep-leakage --fast
```

| Flag | Effect |
|---|---|
| `--tune` / `--n-iter N` | Randomised hyperparameter search per model |
| `--augment-symmetry` | Mirror the training split to enforce antisymmetry (switches CV to `StratifiedGroupKFold`) |
| `--force-augment` | Augment even if swap validation fails — hand-verified mirrors only |
| `--calibrate` | Isotonic calibration on a held-out split, kept only if ECE improves |
| `--calibration-method` | `isotonic` (default) or `sigmoid` |
| `--split temporal` | Train on older campaigns, test on newer ones |
| `--bootstrap N` | Resamples for the lift confidence interval (default 1000, 0 disables) |
| `--campaign-spend` / `--profit-if-correct` / `--opportunity-weight` | Cost matrix inputs |
| `--review-margin` | Expected-cost gap below which a decision needs human review |
| `--models A B` | Restrict the zoo |
| `--fast` | Shrink models for a quick run |
| `--skip-explain` | Skip permutation/SHAP importance |
| `--keep-leakage` | Keep `g1_21`, `g2_21`, `c_28` — demonstration only |

---

## API reference

Base URL: `http://localhost:8000` (or the Cloud Run URL). OpenAPI docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service banner |
| `GET` | `/health` | Liveness/readiness, model load status |
| `GET` | `/model/info` | Model name, version, training date, metrics, feature contract |
| `POST` | `/predict` | Score one comparison |
| `POST` | `/predict/batch` | Score up to 1000 comparisons |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "group_1":    {"g1_1": 1.2, "g1_2": 0.4, "...": "... through g1_20"},
    "group_2":    {"g2_1": 0.7, "g2_2": 0.9, "...": "... through g2_20"},
    "comparison": {"c_1": 0.5,  "c_2": -0.2, "...": "... through c_27"}
  }'
```

```json
{
  "predicted_class": 2,
  "label": "group_2",
  "description": "Group 2 was the most profitable",
  "recommended_action": "Target customer group 2.",
  "confidence": 0.6134,
  "probabilities": {
    "no_group_profitable": 0.1102,
    "group_1": 0.2764,
    "group_2": 0.6134
  }
}
```

All 20 / 20 / 27 keys are required. The schema rejects, with `422` and a reason naming the
offending field:

| Input | Why |
|---|---|
| Missing, unexpected, or post-campaign keys | The contract is exact; a silently ignored `g1_21` would let a caller believe a post-campaign value was used |
| `NaN` | Indistinguishable from `null` downstream but arrives by a different path — usually a failed upstream computation. A caller should have one *deliberate* way to say "no value" |
| `±inf`, or `\|value\| > 1e6` | Finite-but-implausible values pass every type check and overflow during standardisation. Without this the caller receives a `500` for what is unambiguously a client error |
| More than **20%** of the 67 features `null` | The median imputer fills every gap confidently, so a request of 67 nulls would return a well-formed prediction with a confidence score — built entirely from training medians. **A campaign must never be approved from an empty request** |

A `null` value is accepted and imputed **below** that threshold; sparse gaps are what the
imputer is for.

```bash
# Verifiable against the live service:
#   422 - "67 of 67 features are null, above the limit of 13 (20%)"
#   422 - "g1_1 = 1e+308 exceeds the plausible range (|value| <= 1e+06)"
```

Generate a complete example body with:

```python
python -c "
import json
print(json.dumps({
  'group_1':    {f'g1_{i}': 1.0 for i in range(1, 21)},
  'group_2':    {f'g2_{i}': 0.5 for i in range(1, 21)},
  'comparison': {f'c_{i}': 0.25 for i in range(1, 28)},
}))" > payload.json
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @payload.json
```

---

## Design patterns

| Pattern | Implementation | Benefit |
|---|---|---|
| Strategy | `BasePredictor` and subclasses | Swap models without touching the API |
| Factory | `ModelFactory` with a registration decorator | Extend without editing conditionals |
| Singleton | `ModelRegistry` | Artifact deserialised once per process |
| Adapter | `FeatureTransformer` | Model layer stays ignorant of HTTP |
| Dependency Injection | FastAPI `Depends` | Trivial test doubles, no globals in handlers |
| Application Factory | `create_app()` | Isolated app instances per test |

---

## Testing

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

| Job | What it does |
|---|---|
| **data-guard** | Runs `scripts/check_no_data_committed.py`, which refuses a commit whose **staged** changes include the dataset or another blocked path. Runs first — if this fails, no other result matters |
| **lint** | `ruff`, `black --check`, `mypy` |
| **test** | `pytest` with the branch-coverage floor from `pyproject.toml`, optional model libraries installed so the guarded import paths are exercised rather than skipped |
| **security** | `pip-audit` against the **serving** requirements, `bandit -ll` |
| **docker** | Builds the image, starts the container, waits for `/health`, then drives real predictions through it |

**Training is deliberately not in CI.** The dataset is LP-Internal and never committed, so
there is nothing to train on — which is the same constraint that makes the test suite run on
synthetic fixtures. That is what allows the code to be verified by a machine that is not
permitted to see the data.

The container smoke test runs with `ALLOW_BASELINE_FALLBACK=true`, because `artifacts/*.pkl`
is gitignored and CI therefore has no model. The point of the job is that the application
*starts* and answers honestly about its degraded state — a build that succeeds only proves
the image assembles, not that it boots.

### Running tests locally

```bash
pytest                                  # full suite with the coverage floor
pytest tests/test_api.py -v --no-cov    # one module (--no-cov: 30 tests cannot reach 90%)
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

**427 tests across 17 modules, 91.43% branch coverage** against `fail_under = 90` in
`pyproject.toml` — so the gate behaves identically on a laptop and in CI.

Covered: feature engineering arithmetic, leakage removal, payload validation and every
rejection path, predictor strategies, artifact loading and corruption handling, factory
registration, registry semantics and fallback behaviour, evaluation and lift maths,
structured logging, tracking degradation, and all API endpoints including error codes.

Tests use a small synthetic dataset with the real column contract, so the suite is fast,
deterministic, and ships no proprietary data. That is not only convenience — the real
dataset is never committed, so **CI has nothing to train on by design**, and the fixtures are
what make the code verifiable by a machine that is not permitted to see the data.

**Three tests worth singling out**, because they assert properties rather than return values:

- `test_only_the_champion_is_evaluated_on_the_test_set` — fails if `predict(x_test)` is
  reintroduced into the candidate loop. That is how the property was lost the first time, so
  it is now a mechanism rather than a sentence in a report.
- `test_exploration_works_across_separate_single_row_calls` — the service handles **one**
  campaign per request. Every earlier exploration test passed a batch, which is precisely why
  a bug that disabled exploration entirely for single requests survived.
- `TestUnhandledExceptions` — asserts a `500` carries `X-Request-ID` and does **not** echo the
  underlying exception message. Writing it found that the correlation ID was present on every
  successful response and missing on exactly the responses a caller would need it for.

`src/training/train.py` is the weakest module at **72%**: the untested paths are the
calibration branch and MLflow logging, which need a full training run rather than the
`--fast` smoke runs the suite uses. It is included in the measurement — an earlier version
excluded it, which made the headline number flattering.

---

## Deployment (Google Cloud Run)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Backend
gcloud run deploy campaign-api \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated \
  --port 8000 \
  --set-env-vars MODEL_PATH=artifacts/model.pkl,ALLOW_BASELINE_FALLBACK=false

# Frontend, pointed at the backend URL printed above
gcloud run deploy campaign-frontend \
  --source . \
  --region europe-west3 \
  --allow-unauthenticated \
  --port 8501 \
  --set-env-vars API_BASE_URL=https://campaign-api-XXXX.run.app
```

Locally: `docker compose up --build` (API on `:8000`, UI on `:8501`).

### Live deployment

| Service | URL |
|---|---|
| Prediction API | <https://campaign-api-395867964283.europe-west3.run.app> |
| Interactive API docs | <https://campaign-api-395867964283.europe-west3.run.app/docs> |
| Health check | <https://campaign-api-395867964283.europe-west3.run.app/health> |
| Web interface | <https://campaign-frontend-395867964283.europe-west3.run.app> |

Both services run on Cloud Run in `europe-west3`, scale to zero when idle, and are built
from this repository — the API from `Dockerfile`, the frontend from `Dockerfile.frontend`
via `cloudbuild.frontend.yaml`.

The dataset is **not** part of either deployment: `.gcloudignore` excludes `data/`, so the
confidential CSV never leaves the local machine. Only the trained model artifact ships.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `artifacts/model.pkl` | Artifact location |
| `MODEL_TYPE` | `sklearn_pipeline` | Registered predictor name |
| `ALLOW_BASELINE_FALLBACK` | **`false`** | **Fail closed.** See below |
| `AUTOMATION_CONFIDENCE_THRESHOLD` | `0.0` (disabled) | Minimum probability to decide automatically. **Set to `0.60` in deployment** — a fitted parameter, not a constant |
| `EXPLORATION_RATE` | `0.0` (off) | Fraction of decline recommendations overridden so outcomes stay observable |
| `ENABLE_DECISION_LAYER` | `true` | Attach the cost-sensitive recommendation |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `JSON_LOGS` | auto | Structured JSON on Cloud Run (via `K_SERVICE`), plain text locally |
| `ALLOWED_ORIGINS` | `*` | CORS. Credentials are disabled while this is `*` |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL used by the frontend |

**Three defaults are deliberately conservative, and each is a decision rather than a
convention:**

`ALLOW_BASELINE_FALLBACK=false` — a missing or unreadable artifact prevents startup rather
than degrading to the majority-class baseline. This default was **changed** after the
deployed service was found serving the baseline through three consecutive "successful"
deploys: a malformed `--set-env-vars` argument folded two variables into one, `MODEL_PATH`
pointed at a path that did not exist, and a permissive default let the container start anyway
and return confident majority-class predictions. *A fallback that engages by default converts
a loud failure into a quiet wrong answer.* Local development and CI set it `true`
explicitly, because neither has a trained model.

`AUTOMATION_CONFIDENCE_THRESHOLD=0.0` — the gate is **off** in code. 0.60 is a *fitted
parameter* selected on the calibration split for one specific artifact; hard-coding it here
would silently apply one model's operating point to a different model. The deployment sets
it, and it must be re-derived from `findings.json["automation_gate"]["threshold"]` after every
retrain.

`EXPLORATION_RATE=0.0` — exploration spends real budget on knowingly sub-optimal campaigns.
That can be an excellent investment, but shipping `0.05` would quietly commit a marketing team
to misallocating one campaign in twenty. It is a required, explicit input; see
`docs/validation_plan.md` §5.1 for the trade-off. Note that inverse-propensity evaluation is
**undefined** at zero, so this must be set before offline policy evaluation is possible.

---

## Known limitations

- Features are anonymised, so importances are not causally interpretable.
- The lift estimate is offline; the online A/B test in `docs/validation_plan.md` is the
  decisive evidence.
- **Class-0 recall is 8.4%.** The model is weakest exactly where the business value is
  clearest — avoiding spend on a campaign that pays back for neither group.
- **53.8% of predictions FAIL to transform correctly when the two groups are swapped.** Note
  the direction: the symmetry requires 0→0, 1→2, 2→1, so for classes 1 and 2 *changing is
  correct* and staying the same is the violation. The positions are genuinely
  not exchangeable, so using position is legitimate, but the model leans on it heavily and
  will degrade silently if the upstream convention for assigning "group 1" ever changes.
- **No temporal validation is possible** — no time-ordering column exists, so generalisation
  to future campaigns rather than held-out ones is untested.
- **Campaign independence cannot be verified** — no campaign or customer identifier exists,
  so if groups recur across rows every confidence interval here is too narrow.
- The champion's margin over the runner-up is inside noise. Selection happening before the
  test set was opened is the defensible property, not the identity of the winner.
- **The evaluation protocol is a single 20% holdout, not out-of-fold selection with a full
  refit.** That leaves 1,060 development rows unused and estimates accuracy on 1,324 rows
  (standard error 1.37 pp) where 5,296 out-of-fold rows would give roughly 0.68 pp, halving
  the width of the lift interval. Not adopted because it refines a number without changing a
  decision — the lift is already significantly positive and the champion's margin is already
  inside noise. Reasoning in full at `reports/FINAL_REPORT.md` §8.6.
- The frontend exposes 67 numeric inputs; a production version would pull group
  statistics from the customer database instead of manual entry.
