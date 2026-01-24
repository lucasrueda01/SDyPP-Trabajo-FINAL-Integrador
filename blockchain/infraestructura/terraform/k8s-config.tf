resource "kubernetes_config_map_v1" "coordinator_config" {
  metadata {
    name      = "coordinator-config"
    namespace = "blockchain-demo"
  }

  data = {
    REDIS_HOST   = "redis"
    RABBIT_HOST  = "rabbitmq"
    BUCKET_NAME  = google_storage_bucket.blockchain_bucket.name
  }
}

resource "kubernetes_namespace_v1" "blockchain" {
  metadata {
    name = "blockchain-demo"
  }
}