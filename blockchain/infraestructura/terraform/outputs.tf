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

