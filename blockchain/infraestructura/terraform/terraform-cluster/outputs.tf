output "cluster_name" {
  value = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  value = google_container_cluster.primary.endpoint
}

output "nodepool_infra_name" {
  value = google_container_node_pool.infra.name
}

output "nodepool_apps_name" {
  value = google_container_node_pool.apps.name
}

# output "rabbit_ip" {
#   value = google_compute_address.rabbit_ip.address
# }

# output "coordinator_ip" {
#   value = google_compute_address.coordinator_ip.address
# }

# output "pool_manager_ip" {
#   value = google_compute_address.pool_manager_ip.address
# }

output "ingress_ip" {
  value = google_compute_address.ingress_ip.address
}
