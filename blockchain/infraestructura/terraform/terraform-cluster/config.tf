# ConfigMap para el script de inicio de los workers CPU que usa el CPU scaler

resource "kubernetes_config_map_v1" "worker_cpu_script" {
  metadata {
    name      = "worker-cpu-startup"
    namespace = kubernetes_namespace_v1.blockchain_infra.metadata[0].name
  }


  data = {
  "worker_cpu_startup.sh" = file("${path.module}/templates/worker_cpu_startup.sh")
}

}