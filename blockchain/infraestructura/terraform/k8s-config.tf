resource "kubernetes_namespace_v1" "blockchain" {
  metadata {
    name = "blockchain"
  }

  depends_on = [google_container_cluster.primary]
}

resource "kubernetes_config_map_v1" "blockchain_config" {
  metadata {
    name      = "blockchain-config"
    namespace = kubernetes_namespace_v1.blockchain.metadata[0].name
  }

  data = {
    REDIS_HOST  = "redis"
    RABBIT_HOST = "rabbitmq"
    BUCKET_NAME = google_storage_bucket.blockchain_bucket.name
  }

  depends_on = [google_container_cluster.primary]
}
