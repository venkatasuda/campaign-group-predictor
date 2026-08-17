.PHONY: install install-optional train verify-artifact test lint format security api frontend \
        docker-build docker-run deploy verify-deploy \
        tf-init tf-plan tf-apply tf-fmt clean

PY ?= python
IMAGE ?= campaign-group-predictor
REGION ?= europe-west3
SERVICE ?= campaign-api

# Three dependency files, three audiences:
#   requirements.txt          development - notebook, frontend, tests, linters
#   requirements-serve.txt    the API container only - no dev or plotting packages
#   requirements-optional.txt boosted-tree libraries and SHAP, needed to reproduce the
#                             full model zoo but not to serve the scikit-learn champion
install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

# Widens the model zoo from four families to seven and enables SHAP. Every import of
# these is guarded, so the project runs correctly without them - the leaderboard simply
# has fewer rows.
install-optional:
	$(PY) -m pip install -r requirements-optional.txt

# THE canonical training command. Every number in reports/, docs/ and the notebook comes
# from this exact invocation - if it is changed, those documents are wrong until they are
# regenerated.
#
# Why the flags are here rather than in a README:
#
#   --holdout-seed 20260819  Fixes WHICH campaigns are held out, separately from
#                            --random-state (which controls model init and fold assignment).
#                            Omitting it silently falls back to 42 and produces a different
#                            test set - and therefore different accuracy, lift, class-0
#                            recall and symmetry numbers.
#
#   --calibrate              Reserves 20% of the development rows as a calibration split.
#                            This is not optional dressing: it is what makes thresholds and
#                            the calibrator selectable WITHOUT touching the test set. Drop
#                            it and the split becomes 5,296/-/1,324 instead of
#                            4,236/1,060/1,324, every model trains on more data, and the
#                            reported figures shift.
#
# This target exists because that lesson was learned the expensive way: a bare
# `python -m src.training.train --data ... --out artifacts` overwrote the canonical artifact
# with a differently-split experiment, and the deployed service served a model whose
# accuracy did not match a single document describing it. A reproducible run has to be one
# command, not a command plus remembered flags.
#
# Expected result: champion random_forest, split 4,236/1,060/1,324, test accuracy 0.5483,
# lift +8.38 pp, calibration evaluated and REJECTED (6.1% ECE gain against a 10% rule).
# Verify with `make verify-artifact` before deploying or regenerating the notebook.
HOLDOUT_SEED ?= 20260819

train:
	$(PY) -m src.training.train \
		--data data/customerGroups.csv \
		--out artifacts \
		--calibrate \
		--holdout-seed $(HOLDOUT_SEED)

# Asserts the artifact on disk is the one every document describes.
#
# `make train` succeeding proves a model was written, not that it is the right model. This
# is the cheapest check that distinguishes the two, and it belongs before `make deploy`.
verify-artifact:
	@$(PY) -c "import json,sys; \
m=json.load(open('artifacts/metrics.json')); \
c=m['champion_model']; \
got=(m['holdout_seed'], c, round(m['models'][c]['accuracy'],4), round(m['models'][c]['business_lift']['absolute_lift_pp'],2)); \
want=(20260819, 'random_forest', 0.5483, 8.38); \
print('artifact:', got); \
print('expected:', want); \
sys.exit(0 if got==want else 'MISMATCH - the artifact is not the documented run. Re-run `make train`.')"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests frontend scripts
	$(PY) -m black --check src tests frontend scripts
	$(PY) -m mypy

format:
	$(PY) -m black src tests frontend scripts
	$(PY) -m ruff check --fix src tests frontend scripts

# Two questions Dependabot does not answer: do the versions we ship have known CVEs, and
# does our own code contain common security mistakes. Audits the serving dependency set
# rather than the development one, so a finding concerns something that actually reaches
# production.
security:
	$(PY) -m pip_audit --requirement requirements-serve.txt --strict
	$(PY) -m bandit -r src -ll

api:
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

frontend:
	$(PY) -m streamlit run frontend/app.py

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -e ALLOW_BASELINE_FALLBACK=true $(IMAGE)

# Requires: gcloud auth login && gcloud config set project <PROJECT_ID>
#
# Depends on verify-artifact: shipping a model that does not match the documents describing
# it is a worse outcome than not shipping. That is not hypothetical - it happened, and the
# service ran for a while serving a model with a different split, a different seed and an
# accuracy 2.7 points from the reported figure.
#
# AUTOMATION_CONFIDENCE_THRESHOLD is set here, not left to the application default of 0.0.
# The default is 0.0 because the threshold is a fitted parameter belonging to one specific
# artifact; the deployment is the correct place to state which value goes with which model.
deploy: verify-artifact
	gcloud run deploy $(SERVICE) \
		--source . \
		--region $(REGION) \
		--allow-unauthenticated \
		--port 8000 \
		--set-env-vars MODEL_PATH=artifacts/model.pkl,ALLOW_BASELINE_FALLBACK=false,AUTOMATION_CONFIDENCE_THRESHOLD=0.60

# Always run this after `make deploy`.
#
# A Cloud Run deploy reports success once the container answers its health probe - which
# it does even when the model artifact never made it into the image. That is the failure
# this target exists to catch: a green deploy serving no model. It asserts model_loaded is
# literally true and exits non-zero otherwise, so it can gate a release step.
verify-deploy:
	@URL=$$(gcloud run services describe $(SERVICE) --region $(REGION) --format='value(status.url)'); \
	echo "Checking $$URL/health"; \
	curl -fsS "$$URL/health" | tee /dev/stderr | grep -q '"model_loaded": *true' \
		&& echo "OK - model is loaded" \
		|| { echo "FAIL - service is up but no model is loaded"; exit 1; }

# ---------------------------------------------------------------------- infrastructure
# `make deploy` above is the imperative path and is fine for one operator. These targets
# are the declarative one: reviewable in a diff, reproducible across environments, and
# running under a least-privilege service account rather than the default compute identity.
# See terraform/README.md.

tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply

tf-fmt:
	cd terraform && terraform fmt -recursive && terraform validate

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
