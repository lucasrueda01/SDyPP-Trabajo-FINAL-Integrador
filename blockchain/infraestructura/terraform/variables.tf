variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region or zone"
  type        = string
  default     = "us-central1-a"
}
