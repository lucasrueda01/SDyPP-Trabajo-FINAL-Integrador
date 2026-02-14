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

  labels = {
    role       = "worker-cpu"
    managed-by = "terraform"
  }

  network_interface {
    network = "default"
  }

  metadata_startup_script = templatefile(
    "${path.module}/../templates/worker_cpu_startup.sh",
    {
    ingress_ip       = data.terraform_remote_state.cluster.outputs.ingress_ip
  }
  )

  tags = ["worker-cpu"]
}


