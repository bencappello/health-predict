output "ec2_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.main_server.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.main_server.public_dns
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.api_repo.repository_url
} 