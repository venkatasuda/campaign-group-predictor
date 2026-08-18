# Infrastructure for the campaign group predictor.
#
# Two Cloud Run services, one Artifact Registry repository, and one service account per
# service. Everything the `gcloud run deploy` commands in the Makefile create by hand is
# declared here instead, so the environment is reproducible and reviewable in a diff rather
# than reconstructed from shell history.

# ---------------------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------------------
# Enabling services in Terraform makes a fresh project bootstrappable in one apply.
#
# disable_on_destroy is false throughout: disabling an API on `terraform destroy` would
# break every *other* workload in the project that happens to use it. Enabling a service is
# effectively idempotent and free; disabling it is a blast radius.

locals {
  required_apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
  ]
}

data "google_project" "this" {
  project_id = var.project_id
}

# The frontend URL is *derived*, not read from the frontend resource.
#
# Each service needs the other's address: the frontend calls the API, and the API restricts
# CORS to the frontend origin. Referencing both resources from each other is a dependency
# cycle that Terraform refuses to plan. Cloud Run v2 URLs are deterministic
# - https://SERVICE-PROJECTNUMBER.REGION.run.app - so one side of the pair can be computed
# instead of read, which breaks the cycle without introducing a second apply.
locals {
  frontend_url = "https://${var.frontend_service_name}-${data.google_project.this.number}.${var.region}.run.app"
}

resource "google_project_service" "required" {
  for_each = toset(local.required_apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------------------
# Container registry
# ---------------------------------------------------------------------------------------

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  description   = "Container images for the campaign group predictor."
  format        = "DOCKER"
  labels        = var.labels

  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------------------
# Service accounts
# ---------------------------------------------------------------------------------------
# One identity per service, rather than the default compute service account.
#
# The default SA carries broad project-level permissions that neither of these workloads
# needs, and it is shared - so a compromise of the frontend would be a compromise of
# everything else running as that identity. Two named accounts make the blast radius of
# each service its own.

resource "google_service_account" "api" {
  project      = var.project_id
  account_id   = "${var.api_service_name}-sa"
  display_name = "Campaign prediction API"
  description  = "Runtime identity for the prediction service. Deliberately holds no project-level roles."

  depends_on = [google_project_service.required]
}

resource "google_service_account" "frontend" {
  project      = var.project_id
  account_id   = "${var.frontend_service_name}-sa"
  display_name = "Campaign predictor frontend"
  description  = "Runtime identity for the Streamlit interface."

  depends_on = [google_project_service.required]
}

# Both services need to write logs and metrics. Nothing else is granted.
#
# Note what is absent: no Storage, no BigQuery, no Secret Manager. The model artifact is
# baked into the image and no secrets are read at runtime, so granting those roles "just in
# case" would widen the blast radius for no functional gain. When the artifact moves to
# Cloud Storage (see docs/architecture.md), the API account gains exactly
# roles/storage.objectViewer on exactly that bucket - not project-wide.
locals {
  runtime_roles = [
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
  ]

  service_accounts = {
    api      = google_service_account.api.email
    frontend = google_service_account.frontend.email
  }

  runtime_bindings = {
    for pair in setproduct(keys(local.service_accounts), local.runtime_roles) :
    "${pair[0]}-${replace(pair[1], "roles/", "")}" => {
      member = local.service_accounts[pair[0]]
      role   = pair[1]
    }
  }
}

resource "google_project_iam_member" "runtime" {
  for_each = local.runtime_bindings

  project = var.project_id
  role    = each.value.role
  member  = "serviceAccount:${each.value.member}"
}

# ---------------------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  project  = var.project_id
  name     = var.api_service_name
  location = var.region
  labels   = var.labels

  # Traffic is public-facing but arrives through Google's front end; there is no VPC
  # requirement because the service reaches nothing private.
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email

    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }

    # Concurrency is set explicitly, and low, because of what the container holds.
    #
    # Cloud Run defaults to 80 simultaneous requests per instance. That is right for an
    # I/O-bound service that spends its time waiting on a database. This one is CPU-bound:
    # every request runs a 400-tree random forest, and the container is limited to ONE vCPU.
    # Eighty concurrent inferences on one core do not run faster - they queue while
    # contending, and tail latency degrades far more than throughput improves.
    #
    # The value pairs with `n_jobs=1` on the estimator (see src/training/pipeline.py). Those
    # two settings have to be chosen together: `n_jobs=-1` with high concurrency means each
    # request spawns workers for cores that do not exist, and the oversubscription is
    # invisible in single-request benchmarking - which is exactly how it reached production.
    #
    # 8 is a starting point, not a measurement. The honest way to set it is a load test at
    # the target p95, which this project has not run; the number is deliberately
    # conservative so the failure mode is queueing rather than thrashing.
    max_instance_request_concurrency = var.api_max_concurrency

    # Request timeout, set explicitly rather than left at Cloud Run's 300-second default.
    #
    # 300 s is well over two thousand times the measured p95 of 116 ms. Nothing this service does
    # legitimately takes five minutes: the slowest honest request is a 1,000-row batch, which
    # the schema caps precisely so the work stays bounded. A request still running after 60 s
    # is stuck, not slow - and the default holds an instance hostage to it for another four
    # minutes, on a service whose concurrency is deliberately 8. Two such requests remove a
    # quarter of one instance's capacity.
    #
    # Shorter would be defensible for the single-prediction path and wrong for the batch one,
    # so this is sized for the worst legitimate case with headroom, not for the common one.
    timeout = "60s"

    containers {
      image = var.api_image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          cpu    = "1"
          memory = var.api_memory
        }
        # CPU is throttled between requests rather than always allocated. Inference is a
        # short synchronous burst with nothing running in the background, so paying for
        # idle CPU buys nothing.
        cpu_idle = true
      }

      env {
        name  = "MODEL_PATH"
        value = "artifacts/model.pkl"
      }

      env {
        name  = "MODEL_TYPE"
        value = "sklearn_pipeline"
      }

      env {
        name  = "ALLOW_BASELINE_FALLBACK"
        value = tostring(var.allow_baseline_fallback)
      }

      # The operating point the report actually recommends.
      #
      # This is a *fitted parameter tied to one artifact*, not a constant: 0.60 was selected
      # on the held-out calibration split by a stated rule - best accuracy subject to
      # coverage >= 30% - then frozen and applied to the test set once, where it gave 71.58%
      # accuracy on the 34.8% of campaigns above it, against 57.48% for the naive baseline
      # on those same campaigns.
      #
      # It is declared here because a policy that exists only in a report is not deployed.
      # An earlier revision of this service ran with the code default of 0.0, meaning the
      # gate was switched off in production while the report described it as the recommended
      # operating point.
      #
      # RE-DERIVE THIS ON EVERY RETRAIN from findings.json["automation_gate"]["threshold"].
      # A threshold chosen for one champion is meaningless for another.
      env {
        name  = "AUTOMATION_CONFIDENCE_THRESHOLD"
        value = tostring(var.automation_confidence_threshold)
      }

      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }

      # Lock CORS to the deployed frontend. The application disables credentials whenever
      # this is "*", because wildcard-plus-credentials is invalid per the CORS spec.
      # Derived rather than referenced - see the note on `local.frontend_url` above.
      env {
        name  = "ALLOWED_ORIGINS"
        value = local.frontend_url
      }

      # Three endpoints, three questions, and each probe asks the one it can act on.
      #
      # Both probes previously pointed at /health, which answers a fourth question - "what is
      # the detailed status, for a human?" - and answers it with 200 even when the service is
      # degraded, because a dashboard needs to read the body. A probe reads only the status
      # code. So a container serving the 46%-accurate majority-class baseline reported itself
      # healthy, was given traffic, and nothing in the infrastructure disagreed.
      #
      # The endpoints to fix that were built and tested; the infrastructure just never used
      # them. Implementing the right probe and then not wiring it in leaves the original
      # failure fully intact while every review of the code says it was handled.
      #
      # Startup -> /ready: 503 until the artifact is deserialised AND is not the baseline, so
      # Cloud Run withholds traffic from a container that has booted but cannot yet serve a
      # real prediction. failure_threshold x period_seconds = 30s, comfortably above the
      # observed model load.
      startup_probe {
        http_get {
          path = "/ready"
          port = 8000
        }
        initial_delay_seconds = 5
        period_seconds        = 5
        failure_threshold     = 6
      }

      # Liveness -> /live: process responsiveness only, deliberately checking nothing about
      # the model. A liveness probe that also checked the artifact would restart a container
      # whose only problem is a missing model - a crash loop that cannot fix itself, because
      # the replacement container is missing the same file.
      liveness_probe {
        http_get {
          path = "/live"
          port = 8000
        }
        period_seconds = 30
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "frontend" {
  project  = var.project_id
  name     = var.frontend_service_name
  location = var.region
  labels   = var.labels

  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.frontend.email

    scaling {
      min_instance_count = 0
      max_instance_count = 4
    }

    containers {
      image = var.frontend_image

      ports {
        container_port = 8501
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        cpu_idle = true
      }

      # The frontend holds no business logic; it calls the API. Pointing it at the API's
      # own URL rather than a hardcoded string means renaming the service cannot leave the
      # two out of sync.
      env {
        name  = "API_BASE_URL"
        value = google_cloud_run_v2_service.api.uri
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }

  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------------------
# Public access
# ---------------------------------------------------------------------------------------
# `allUsers` is how Cloud Run expresses "no authentication required". It is gated behind a
# variable and defaults on only because reviewers need to open the URLs; in production this
# is false and an API Gateway or IAP sits in front.

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "frontend_public" {
  count = var.allow_unauthenticated ? 1 : 0

  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# When the services are private, the frontend still has to reach the API. Granting the
# frontend's identity the invoker role on the API - and nothing else - is what makes
# "private by default" workable rather than something that gets switched off in a hurry.
resource "google_cloud_run_v2_service_iam_member" "frontend_calls_api" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.frontend.email}"
}
