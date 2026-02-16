resource "google_container_node_pool" "infra" {
  name     = "${var.cluster_name}-infra-pool"
  cluster = google_container_cluster.primary.name
  location = var.zone
  initial_node_count = var.infra_initial_nodes

  node_config {
    machine_type = "e2-medium"

    disk_type    = "pd-standard"
    disk_size_gb = 30

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      node-role = "infra"
    }
  }

  autoscaling {
    min_node_count = var.infra_min_nodes
    max_node_count = var.infra_max_nodes
  }
}
