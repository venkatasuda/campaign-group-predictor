# Infrastructure as code

Terraform for the two Cloud Run services, the container registry and the runtime
identities. Everything the `gcloud` commands in the root `Makefile` create by hand is
declared here instead.

## Why this exists

`make deploy` works, and for a single-operator demo it is enough. It does not survive
contact with a team:

| `gcloud run deploy` | Terraform |
|---|---|
| Configuration lives in shell history | Configuration lives in a diff |
| Drift is invisible until something breaks | `terraform plan` shows drift before you apply |
| A second environment is a copy-pasted command | A second environment is a `.tfvars` file |
| Rollback is remembering what the flags were | Rollback is `git revert` |
| Runs as the default compute service account | Runs as a named least-privilege identity |

The last row is the one that matters most here. The `gcloud` path silently uses the default
compute service account, which carries broad project-level permissions neither service
needs and is shared with everything else in the project. This configuration creates one
identity per service holding only `logging.logWriter` and `monitoring.metricWriter`.

## Status of this configuration

Stated plainly, because infrastructure code should say how far it has actually been taken.

| | |
|---|---|
| `terraform fmt -recursive` | ✅ Clean |
| `terraform init` | ✅ Provider `hashicorp/google` v7.44.0 |
| `terraform validate` | ✅ **Success — the configuration is valid** |
| `terraform plan` / `apply` | ❌ Not run against a live project |

The two services described here are currently deployed with the `gcloud` commands in the
root `Makefile`. This configuration is the declarative equivalent and has not yet replaced
them; adopting it against the existing services requires `terraform import` first, so that
Terraform adopts them rather than attempting to recreate resources that already exist.

### A note on the lockfile

`.terraform.lock.hcl` currently records checksums for `windows_amd64` only, because the
provider was installed from a local directory rather than the registry (network
restrictions on the development machine blocked the registry download). CI runs on Linux
and would fail to install from this lockfile as it stands.

The fix is one command, and it needs registry access:

```bash
terraform providers lock -platform=linux_amd64 -platform=darwin_arm64 -platform=windows_amd64
```

This is exactly the class of problem a lockfile exists to surface — a dependency that
resolves on one engineer's machine and not in CI. Recording it here rather than deleting
the lockfile is the point.

## Usage

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars    # then edit
terraform init
terraform plan          # read this before applying
terraform apply
```

> Applying against the already-deployed services will fail on resources that exist. Import
> them first:
>
> ```bash
> terraform import google_cloud_run_v2_service.api \
>   projects/PROJECT/locations/europe-west3/services/campaign-api
> ```

Images must exist before applying. Build and push first:

```bash
REPO="europe-west3-docker.pkg.dev/$PROJECT_ID/campaign-predictor"
gcloud auth configure-docker europe-west3-docker.pkg.dev

docker build -t "$REPO/api:v1.0.0" .
docker build -t "$REPO/frontend:v1.0.0" -f Dockerfile.frontend .
docker push "$REPO/api:v1.0.0"
docker push "$REPO/frontend:v1.0.0"
```

Then verify the deploy actually serves a model — Cloud Run reports success as soon as the
container answers its health probe, which it does even with no artifact:

```bash
curl -s "$(terraform output -raw api_health_url)" | grep -q '"model_loaded": *true' \
  && echo OK || echo "FAIL - service is up but serving no model"
```

## Design decisions

**Remote state is configured but commented out.** A GCS backend cannot be assumed to exist
in a fresh project, and this challenge has one operator. For a team it is not optional:
local state means no locking, so two concurrent applies can corrupt each other, and there
is no history to roll back to. The bucket must have object versioning enabled.

**Provider versions are constrained loosely and pinned in `.terraform.lock.hcl`.** The
constraint says what the configuration is compatible with; the lockfile — committed —
guarantees every engineer and CI run resolves the same build.

**APIs are enabled here, with `disable_on_destroy = false`.** A fresh project becomes
bootstrappable in one apply. Disabling an API on destroy would break every other workload in
the project that happens to use it.

**The frontend URL is derived, not referenced.** Each service needs the other's address:
the frontend calls the API, the API restricts CORS to the frontend origin. Referencing both
resources from each other is a dependency cycle Terraform refuses to plan. Cloud Run v2
URLs are deterministic, so one side is computed from the project number instead — which
breaks the cycle without needing two applies.

**`min_instance_count = 0`.** Campaign planning is bursty, not continuous, so idle cost
would dominate. The trade is cold-start latency, and a cold start here pays model
deserialisation. Raise it only if that becomes a measured complaint rather than a
theoretical one.

**Startup probes as well as liveness probes.** Cloud Run must not route traffic to a
container that has started but has not finished loading the model. Without a startup probe,
the first requests after a scale-up fail in a way that never reproduces locally.

**`cpu_idle = true`.** Inference is a short synchronous burst with nothing running in the
background. Always-allocated CPU would be paid for and unused.

## Not in scope

Deliberately absent, matching the boundary stated in `docs/architecture.md`:

- **Cloud Storage for artifacts** — the model is baked into the image. Adequate for one
  model. When it moves, the API's service account gains `roles/storage.objectViewer` on
  exactly that bucket, not project-wide.
- **BigQuery** — the retraining loop is a design, not a running pipeline.
- **Monitoring alert policies** — drift reference distributions are captured and PSI is
  computed, but nothing pages anyone yet.
- **Custom domain and Cloud Armor** — the run.app URLs are sufficient for review.

These are scope decisions, not oversights. Each is cheap to add precisely because the seams
already exist.
