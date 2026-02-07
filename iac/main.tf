# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
}

# Random suffix for S3 bucket to ensure uniqueness
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Data source for AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
  owners = ["099720109477"] # Canonical
}

# 1. VPC, Public Subnet, Internet Gateway
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name = "health-predict-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = data.aws_availability_zones.available.names[0]
  tags = {
    Name = "health-predict-public-subnet"
  }
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "health-predict-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }
  tags = {
    Name = "health-predict-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

# 2. Security Group
resource "aws_security_group" "ec2_sg" {
  name        = "health-predict-ec2-sg"
  description = "Allow SSH, Airflow, MLflow, and API access"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH from your_ip"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    description = "Airflow UI from your_ip"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    description = "MLflow UI from your_ip"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    description = "API from your_ip (adjust port if needed)"
    from_port   = 8000 # Assuming API runs on 8000, adjust as per your API
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  ingress {
    description = "Monitoring Dashboard (Streamlit)"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["${var.your_ip}/32"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1" # Allow all outbound traffic
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "health-predict-ec2-sg"
  }
}

# 3. IAM Role for EC2
resource "aws_iam_role" "ec2_role" {
  name = "health-predict-ec2-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
  tags = {
    Name = "health-predict-ec2-role"
  }
}

resource "aws_iam_policy" "s3_access" {
  name        = "health-predict-s3-access-policy"
  description = "Policy for S3 bucket access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.ml_artifacts.arn,
          "${aws_s3_bucket.ml_artifacts.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "ecr_access" {
  name        = "health-predict-ecr-access-policy"
  description = "Policy for ECR access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:GetRepositoryPolicy",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage"
        ]
        Effect   = "Allow"
        Resource = aws_ecr_repository.api_repo.arn
      },
      {
        Action   = "ecr:GetAuthorizationToken" # Required for Docker login to ECR
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.s3_access.arn
}

resource "aws_iam_role_policy_attachment" "ecr_attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ecr_access.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "health-predict-ec2-profile"
  role = aws_iam_role.ec2_role.name
}

# 4. EC2 Instance
resource "aws_instance" "main_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name
  key_name               = "ik-keys" # IMPORTANT: Replace with your EC2 key pair name

  user_data = <<-EOF
              #!/bin/bash
              sudo apt-get update -y
              sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
              # Install Docker
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
              sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
              sudo apt-get update -y
              sudo apt-get install -y docker-ce docker-ce-cli containerd.io
              sudo usermod -aG docker ubuntu
              # Install Docker Compose
              sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
              sudo chmod +x /usr/local/bin/docker-compose
              # Install Git
              sudo apt-get install -y git
              EOF

  tags = {
    Name = "health-predict-main-server"
  }

  # Ensure IAM profile is created before instance
  depends_on = [aws_iam_instance_profile.ec2_profile]
}

# 5. S3 Bucket
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.bucket_name_prefix}-${random_id.bucket_suffix.hex}"
  tags = {
    Name = "health-predict-ml-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "ml_artifacts_versioning" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "ml_artifacts_access_block" {
  bucket                  = aws_s3_bucket.ml_artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 6. ECR Repository
resource "aws_ecr_repository" "api_repo" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "health-predict-api-repo"
  }
}

# Data source for availability zones
data "aws_availability_zones" "available" {} 