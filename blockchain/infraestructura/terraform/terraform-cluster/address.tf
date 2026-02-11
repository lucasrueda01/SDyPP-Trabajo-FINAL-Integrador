resource "google_compute_address" "rabbit_ip" {
  name   = "rabbit-ip"
  region = var.region
}

resource "google_compute_address" "coordinator_ip" {
  name   = "coordinator-ip"
  region = var.region
}

resource "google_compute_address" "pool_manager_ip" {
  name   = "pool-manager-ip"
  region = var.region
}
