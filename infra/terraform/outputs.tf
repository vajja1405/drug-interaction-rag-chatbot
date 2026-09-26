output "cluster_name" { value = module.eks.cluster_name }
output "cluster_endpoint" { value = module.eks.cluster_endpoint }
output "ecr_repository" { value = aws_ecr_repository.api.repository_url }
output "kubeconfig_cmd" { value = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}" }
