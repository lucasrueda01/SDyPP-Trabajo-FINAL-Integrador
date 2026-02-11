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

resource "kubernetes_config_map_v1" "worker_cpu_script" {
  metadata {
    name      = "worker-cpu-startup"
    namespace = kubernetes_namespace_v1.blockchain_infra.metadata[0].name
  }


  data = {
    "worker_cpu_startup.sh" = file("${path.module}/templates/worker_cpu_startup.sh")
  }
}
