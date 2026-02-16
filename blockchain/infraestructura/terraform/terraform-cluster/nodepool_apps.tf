resource "google_container_node_pool" "apps" {
  name       = "${var.cluster_name}-apps-pool"
  cluster    = google_container_cluster.primary.name
  location   = var.zone
  initial_node_count = var.apps_initial_nodes

  node_config {
    machine_type = var.apps_machine_type

    disk_type = "pd-standard"
    disk_size_gb = 30

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      node-role = "apps"
    }
  }

  autoscaling {
    min_node_count = var.apps_min_nodes
    max_node_count = var.apps_max_nodes
  }
}
