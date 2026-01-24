output "bucket_name" {
  value = google_storage_bucket.blockchain_bucket.name
}

output "cluster_name" {
  value = google_container_cluster.demo.name
}
