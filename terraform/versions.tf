# Provider and version constraints.
#
# Versions are constrained loosely here and pinned exactly in `.terraform.lock.hcl`, which
# is committed. That split is deliberate: the constraint expresses what the configuration
# is *compatible* with, while the lockfile guarantees every engineer and every CI run
# resolves the same provider build. Pinning an exact version here instead would force a
# config edit for a patch bump and still not guarantee reproducibility, because module
# dependencies resolve independently.

terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }

  # Remote state, commented out because this challenge is a single-operator project and a
  # bucket cannot be assumed to exist.
  #
  # For anything with more than one operator this is not optional. Local state means no
  # locking (two applies can corrupt each other), no history, and state living on one
  # laptop - and Terraform state contains resource metadata that should not sit in a
  # working directory. The bucket must have versioning enabled so a corrupted state can be
  # rolled back.
  #
  # backend "gcs" {
  #   bucket = "campaign-predictor-tfstate"
  #   prefix = "prod"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
