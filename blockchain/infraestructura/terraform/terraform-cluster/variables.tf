variable "project_id" {
  type        = string
  description = "GCP Project ID"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "zone"   { default = "us-central1-a" }

variable "cluster_name" { default = "cluster-blockchain" }

# node pool sizes / machines
variable "infra_machine_type" { default = "e2-medium" }
variable "infra_initial_nodes" { default = 1 }

variable "apps_machine_type" { default = "e2-medium" }
variable "apps_initial_nodes" { default = 1 }
variable "apps_min_nodes" { default = 1 }
variable "apps_max_nodes" { default = 3 }
variable "infra_min_nodes" { default = 3 }
variable "infra_max_nodes" { default = 3 }