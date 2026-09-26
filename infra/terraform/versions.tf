terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.70" }
  }
  # Remote state for team use; local state is fine for a personal sandbox.
  # backend "s3" { bucket = "drug-ai-tfstate" key = "eks/terraform.tfstate" region = "us-east-1" }
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "drug-interaction-ai", managed_by = "terraform" } }
}
