terraform {
  backend "gcs" {
    bucket  = "tf-state-blockchain"
    prefix  = "workers/state"
  }
}
