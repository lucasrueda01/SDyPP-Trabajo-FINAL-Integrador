resource "kubernetes_namespace_v1" "blockchain_infra" {
  metadata {
    name = "blockchain-infra"
  }

  depends_on = [google_container_cluster.primary]
}

resource "kubernetes_namespace_v1" "blockchain_apps" {
  metadata {
    name = "blockchain-apps"
  }

  depends_on = [google_container_cluster.primary]
}


