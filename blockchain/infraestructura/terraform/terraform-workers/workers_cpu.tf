data "terraform_remote_state" "cluster" {
  backend = "gcs"

  config = {
    bucket = "tf-state-blockchain"   # tu bucket real
    prefix = "cluster/state"         # el mismo prefix que usaste en cluster
  }
}

resource "google_compute_instance" "worker_cpu" {
  count        = var.worker_cpu_count
  name         = "${var.cluster_name}-worker-cpu-${count.index}"
  machine_type = var.worker_machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
  }

  metadata_startup_script = templatefile(
    "${path.module}/../templates/worker_cpu_startup.sh",
    {
    rabbit_host       = data.terraform_remote_state.cluster.outputs.rabbit_ip
    coordinator_host  = data.terraform_remote_state.cluster.outputs.coordinator_ip
    pool_manager_host = data.terraform_remote_state.cluster.outputs.pool_manager_ip
  }
  )

  tags = ["worker-cpu"]
}


