resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  remove_default_node_pool = true
  initial_node_count       = 1

  node_config {
    disk_type    = "pd-standard"
    disk_size_gb = 30
  }

  deletion_protection = false

  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {}

  addons_config {
    http_load_balancing {
      disabled = false
    }
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
  }

  monitoring_config {
    managed_prometheus {
      enabled = false
    }

    component_config {
    enable_components = []
  }
}


}
