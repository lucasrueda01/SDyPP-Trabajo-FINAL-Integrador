variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" { default = "us-central1" }
variable "zone"   { default = "us-central1-a" }

variable "cluster_name" { default = "cluster-blockchain" }

# worker VMs
variable "worker_cpu_count" { default = 2 }
variable "worker_machine_type" { default = "e2-small" }


