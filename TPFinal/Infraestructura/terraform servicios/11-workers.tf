resource "google_compute_instance_template" "worker_template" {
  name         = "worker-template"
  machine_type = var.tipo_vm
  region       = var.region

  disk {
    auto_delete = true
    boot        = true
    source_image = var.imagen
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    ssh-keys = "${split("@", data.google_client_openid_userinfo.me.email)[0]}:${tls_private_key.ssh_key.public_key_openssh}"
  }
}

resource "google_compute_instance_group_manager" "worker_group" {
  name               = "worker-group"
  zone               = var.zone
  base_instance_name = "worker"
  version {
    instance_template = google_compute_instance_template.worker_template.self_link
  }
  target_size = 1
}

resource "google_compute_autoscaler" "worker_autoscaler" {
  name   = "worker-autoscaler"
  zone   = var.zone
  target = google_compute_instance_group_manager.worker_group.id

  autoscaling_policy {
    max_replicas    = var.max_replicas
    min_replicas    = var.min_replicas
    cooldown_period = 60

    cpu_utilization {
      target = 0.6
    }
  }
}