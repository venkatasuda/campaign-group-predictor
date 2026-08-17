output "api_url" {
  description = "Prediction API base URL."
  value       = google_cloud_run_v2_service.api.uri
}

output "api_docs_url" {
  description = "Interactive OpenAPI documentation."
  value       = "${google_cloud_run_v2_service.api.uri}/docs"
}

output "api_health_url" {
  description = "Health endpoint. Assert model_loaded is true after every deploy."
  value       = "${google_cloud_run_v2_service.api.uri}/health"
}

output "frontend_url" {
  description = "Streamlit interface."
  value       = google_cloud_run_v2_service.frontend.uri
}

output "artifact_registry" {
  description = "Docker repository to push images to before applying."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}

output "service_accounts" {
  description = "Runtime identities, for auditing what each service is permitted to do."
  value = {
    api      = google_service_account.api.email
    frontend = google_service_account.frontend.email
  }
}

# Surfaced as an output rather than left implicit, because it is the setting most likely to
# be wrong in a real deployment and the least likely to announce itself: a service running
# with the fallback enabled looks healthy while serving a baseline.
output "baseline_fallback_enabled" {
  description = "True means the API may serve the majority-class baseline if the artifact fails to load. Should be false outside development."
  value       = var.allow_baseline_fallback
}
