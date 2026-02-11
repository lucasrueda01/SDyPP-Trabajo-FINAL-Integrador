resource "google_storage_bucket" "blockchain_bucket" {
  name     = "bucket-integrador-${var.project_id}"
  location = "US"

  storage_class = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy = true
}

terraform {
  backend "gcs" {
    bucket  = "tf-state-blockchain"
    prefix  = "cluster/state"
  }
}
