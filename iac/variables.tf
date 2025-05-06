variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.micro" # Free tier eligible
}

variable "bucket_name_prefix" {
  description = "Prefix for the S3 bucket name. A random suffix will be added."
  type        = string
  default     = "health-predict-mlops"
}

variable "ecr_repository_name" {
  description = "Name for the ECR repository"
  type        = string
  default     = "health-predict-api"
}

variable "your_ip" {
  description = "Your public IP address to allow SSH access to the EC2 instance. Get it from https://checkip.amazonaws.com/"
  type        = string
  # Sensitive = true # Uncomment if you don't want this to be shown in logs, but you'll need to provide it at apply time.
  default     = "23.241.28.205"
} 