resource "google_compute_router" "worker_nat_router" {
  name    = "worker-cpu-router"
  network = "default"
  region  = var.region
}

resource "google_compute_router_nat" "worker_nat" {
  name                               = "worker-cpu-nat"
  router                            = google_compute_router.worker_nat_router.name
  region                            = var.region
  nat_ip_allocate_option            = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = false
    filter = "ERRORS_ONLY"
  }
}
