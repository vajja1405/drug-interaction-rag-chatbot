variable "region" {
  type    = string
  default = "us-east-1"
}

variable "name" {
  type    = string
  default = "drug-ai"
}

variable "cluster_version" {
  type    = string
  default = "1.31"
}

variable "vpc_cidr" {
  type    = string
  default = "10.40.0.0/16"
}

variable "gpu_node_count" {
  description = "GPU nodes for vLLM. Keep 0 unless benchmarking: a g5.xlarge costs about $1/hour."
  type        = number
  default     = 0
}
