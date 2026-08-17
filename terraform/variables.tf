variable "project_id" {
  description = "GCP project ID. No default - an accidental apply into the wrong project is expensive."
  type        = string
}

variable "region" {
  description = "Deployment region. europe-west3 (Frankfurt) keeps campaign data in the EU."
  type        = string
  default     = "europe-west3"
}

variable "api_service_name" {
  description = "Cloud Run service name for the prediction API."
  type        = string
  default     = "campaign-api"
}

variable "frontend_service_name" {
  description = "Cloud Run service name for the Streamlit interface."
  type        = string
  default     = "campaign-frontend"
}

variable "repository_id" {
  description = "Artifact Registry repository holding both container images."
  type        = string
  default     = "campaign-predictor"
}

variable "api_image" {
  description = <<-EOT
    Fully qualified API image, e.g.
    europe-west3-docker.pkg.dev/PROJECT/campaign-predictor/api:abc1234.

    Prefer an immutable digest or commit SHA over `:latest`. A mutable tag makes the
    deployed revision unknowable from the state file, which defeats the purpose of
    declaring it here.
  EOT
  type        = string
}

variable "frontend_image" {
  description = "Fully qualified frontend image. Same tagging guidance as api_image."
  type        = string
}

variable "allow_unauthenticated" {
  description = <<-EOT
    Whether the services accept unauthenticated requests.

    True for this challenge so reviewers can open the URLs. In production this should be
    false, with an API Gateway or IAP in front, because the service authorises campaign
    spend and its request payload describes customer segments.
  EOT
  type        = bool
  default     = true
}

variable "allow_baseline_fallback" {
  description = <<-EOT
    Whether the API may start with the majority-class baseline when the model artifact
    cannot be loaded.

    Deliberately false here. A service that reports healthy while silently serving a
    baseline is the worst available failure mode: monitoring stays green, campaign budget
    gets allocated by a coin flip weighted to the majority class, and nobody finds out
    until someone audits the outcomes.

    This is not hypothetical. The deployed service ran in exactly that state through three
    consecutive successful deploys, because a malformed --set-env-vars argument folded two
    variables into one and MODEL_PATH pointed at a path that did not exist. Declaring the
    value here rather than passing it on a command line is precisely what stops that class
    of mistake: an argument-quoting bug cannot survive a reviewed diff.
  EOT
  type        = bool
  default     = false
}

variable "api_max_concurrency" {
  description = <<-EOT
    Simultaneous requests one API instance will accept.

    Cloud Run's default is 80, which suits an I/O-bound service waiting on a database. This
    service is CPU-bound - every request runs a 400-tree random forest - and the container
    has one vCPU. High concurrency on one core does not increase throughput; it queues
    requests while they contend, and tail latency suffers far more than throughput gains.

    Must be chosen together with the estimator's n_jobs (set to 1 in
    src/training/pipeline.py). n_jobs=-1 combined with high concurrency spawns workers for
    cores that do not exist, and the oversubscription does not appear in single-request
    benchmarking.

    8 is conservative rather than measured. Setting it properly requires a load test at the
    target p95, which this project has not run - so the value errs toward queueing rather
    than thrashing, and is recorded as an assumption rather than a result.
  EOT
  type        = number
  default     = 8

  validation {
    condition     = var.api_max_concurrency >= 1 && var.api_max_concurrency <= 80
    error_message = "Concurrency must be between 1 and 80 (the Cloud Run maximum)."
  }
}

variable "automation_confidence_threshold" {
  description = <<-EOT
    Minimum predicted probability required to decide a campaign automatically. Below it the
    response is flagged review_required and routed to a person.

    0.60 is a FITTED PARAMETER for one specific artifact, not a constant. It was selected on
    the held-out calibration split by a stated rule - best accuracy subject to coverage of
    at least 30% - then frozen and applied to the test set once: 71.58% accuracy on the
    34.8% of campaigns above the threshold, against 57.48% for the naive baseline on those
    same campaigns.

    Re-derive it from findings.json["automation_gate"]["threshold"] after every retrain. A
    threshold chosen for one champion is meaningless for another, which is why the
    application default is 0.0 (gate disabled) rather than this value hard-coded in code.
  EOT
  type        = number
  default     = 0.60

  validation {
    condition     = var.automation_confidence_threshold >= 0.0 && var.automation_confidence_threshold < 1.0
    error_message = "Must lie in [0, 1). A threshold of 1.0 would escalate every campaign."
  }
}

variable "api_min_instances" {
  description = <<-EOT
    Minimum API instances. Zero enables scale-to-zero.

    Campaign planning is bursty rather than continuous, so idle cost dominates if this is
    raised.

    The cold start has been measured rather than assumed: **15.5 seconds** for the first
    request after an idle period, against a steady-state p95 of 91 ms
    (`reports/latency.json`). That is container start, image pull and model
    deserialisation together.

    Zero remains the right default here - one person absorbs that wait once per planning
    session, and the alternative is paying for an always-warm instance between campaigns.
    Raise it to 1 if either becomes true: the API acquires a machine consumer that retries
    on timeout, or the frontend's first-load experience is judged unacceptable. Note the
    measurement predates slimming the serving image, so the true figure is now lower.
  EOT
  type        = number
  default     = 0
}

variable "api_max_instances" {
  description = "Maximum API instances. A ceiling, not a target - it bounds cost under a traffic spike."
  type        = number
  default     = 10
}

variable "api_memory" {
  description = "Memory per API instance. scikit-learn plus the pipeline fits comfortably in 1Gi."
  type        = string
  default     = "1Gi"
}

variable "labels" {
  description = "Labels applied to every resource, so cost and ownership are attributable."
  type        = map(string)
  default = {
    application = "campaign-group-predictor"
    managed-by  = "terraform"
    environment = "demo"
  }
}
