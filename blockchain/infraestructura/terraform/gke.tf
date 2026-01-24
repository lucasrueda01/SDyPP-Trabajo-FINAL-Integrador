resource "google_container_cluster" "demo" {
  name     = "blockchain-demo"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1
}

resource "google_container_node_pool" "primary" {
  name     = "primary-nodes"
  cluster  = google_container_cluster.demo.name
  location = var.region

  node_count = 1

  node_config {
    machine_type = "e2-small"
    disk_size_gb = 30
    disk_type    = "pd-standard"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}
