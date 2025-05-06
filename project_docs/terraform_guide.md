# Terraform Infrastructure Setup Guide for Health Predict MLOps

This guide provides step-by-step instructions to deploy the necessary AWS infrastructure for the Health Predict MLOps project using the provided Terraform scripts.

**Prerequisites:**

1.  **AWS Account & CLI Configured:** Ensure you have an AWS account and the AWS CLI installed and configured with credentials that have sufficient permissions to create the resources defined (VPC, EC2, S3, ECR, IAM).
2.  **Terraform Installed:** Install Terraform on your local machine. You can download it from the [official Terraform website](https://www.terraform.io/downloads.html).
3.  **EC2 Key Pair:** You need an EC2 Key Pair in the AWS region you intend to deploy the resources. You will need to provide the name of this key pair to the Terraform configuration. If you don't have one, create it in the EC2 console.
4.  **Your Public IP:** You'll need your public IP address to restrict SSH and other access to the EC2 instance. You can find this by visiting [https://checkip.amazonaws.com/](https://checkip.amazonaws.com/).

**Step 1: Review and Customize Terraform Configuration**

Navigate to the `iac/` directory in your project.

1.  **`variables.tf`:**
    *   `aws_region`: Review the default AWS region (`us-east-1`). Change it if you prefer a different region. Ensure your chosen EC2 Key Pair exists in this region.
    *   `instance_type`: Defaults to `t2.micro` (Free Tier eligible). You can change this if needed, but be mindful of costs.
    *   `your_ip`: **This is a critical variable.** You *must* set this to your public IP address to allow SSH, Airflow, and MLflow access to the EC2 instance. The current placeholder will likely prevent you from accessing the instance. Find your IP (e.g., from [https://checkip.amazonaws.com/](https://checkip.amazonaws.com/)) and update the `default` value or provide it when prompted during `terraform apply`.
        ```terraform
        variable "your_ip" {
          description = "Your public IP address to allow SSH access to the EC2 instance. Get it from https://checkip.amazonaws.com/"
          type        = string
          default     = "YOUR_IP_ADDRESS_HERE" # <<< --- UPDATE THIS
        }
        ```
    *   `bucket_name_prefix` and `ecr_repository_name`: These have default values but can be changed if desired.

2.  **`main.tf`:**
    *   **EC2 `key_name`**: Locate the `aws_instance` resource block named `main_server`. You **must** replace `"your-ec2-key-pair-name"` with the actual name of your EC2 key pair.
        ```terraform
        resource "aws_instance" "main_server" {
          # ... other configurations ...
          key_name               = "your-ec2-key-pair-name" # <<< --- REPLACE THIS
          # ... other configurations ...
        }
        ```
    *   Review other resource configurations if you have specific needs (e.g., CIDR blocks, port numbers for security groups if your services will run on different ports).

**Step 2: Initialize Terraform**

Once you've customized the files, open your terminal, navigate to the `iac/` directory, and run:

```bash
cd iac
terraform init
```

This command initializes the Terraform working directory, downloading the necessary provider plugins (in this case, for AWS).

**Step 3: Plan the Deployment**

After initialization, create an execution plan:

```bash
terraform plan -out=tfplan
```

This command shows you what resources Terraform will create, modify, or destroy. Review the plan carefully to ensure it matches your expectations. The `-out=tfplan` saves the plan to a file, which is a good practice.

If you did not set a default for `your_ip` and it's not marked sensitive, Terraform will prompt you for it here. You can also use a `terraform.tfvars` file or supply variables via the command line, e.g. `terraform plan -var="your_ip=1.2.3.4"`

**Step 4: Apply the Configuration (Deploy Infrastructure)**

If the plan looks good, apply the configuration to create the resources in AWS:

```bash
terraform apply tfplan
```

Terraform will again show you the plan and ask for confirmation before proceeding. Type `yes` and press Enter.

This process can take several minutes as AWS provisions the resources.

**Step 5: Verify Resource Creation & Get Outputs**

Once `terraform apply` completes successfully:

1.  **AWS Console:** Log in to your AWS Management Console and navigate to the respective services (VPC, EC2, S3, ECR, IAM) to verify that the resources have been created as expected in the region you specified.
2.  **Terraform Outputs:** Terraform will display the defined outputs. You can also retrieve them at any time using:
    ```bash
    terraform output
    ```
    The key outputs are:
    *   `ec2_public_ip`: The public IP address of your newly created EC2 instance. You'll use this to SSH into the machine.
    *   `ec2_public_dns`: The public DNS name of the EC2 instance.
    *   `s3_bucket_name`: The name of the S3 bucket created for ML artifacts.
    *   `ecr_repository_url`: The URL for your ECR repository.

    Save these values, as you will need them for subsequent project steps (e.g., SSHing into EC2, configuring MLflow, pushing Docker images).

**Step 6: Accessing Your EC2 Instance**

Once the instance is running (check the EC2 console), you can SSH into it using its public IP and your EC2 key pair:

```bash
ssh -i /path/to/your/key-pair-name.pem ubuntu@<EC2_PUBLIC_IP>
```

Replace `/path/to/your/key-pair-name.pem` with the actual path to your private key file and `<EC2_PUBLIC_IP>` with the IP address from the Terraform output.

Inside the EC2 instance, you can verify that Docker, Docker Compose, and Git were installed by the user data script (e.g., `docker --version`, `docker-compose --version`, `git --version`).

**CRITICAL: Managing Costs - Teardown**

AWS resources incur costs when they are running. To avoid unexpected charges, it is **essential** to destroy the infrastructure when you are not actively using it or when you have finished with the project.

To destroy all resources managed by this Terraform configuration, run the following command from the `iac/` directory:

```bash
terraform destroy
```

Terraform will show you all the resources it intends to destroy and will ask for confirmation. Type `yes` and press Enter.

**ALWAYS REMEMBER TO RUN `terraform destroy` WHEN YOU ARE DONE WORKING TO PREVENT ONGOING CHARGES.**

Also, make sure to stop any EC2 instances manually if you are taking a break and haven't run `terraform destroy`, though destroying and reapplying is the cleaner IaC approach for non-production environments.

**Troubleshooting:**

*   **Errors during `terraform init`:** Check your internet connection and ensure Terraform can access the HashiCorp plugin repository.
*   **Errors during `terraform plan` or `apply`:** Carefully read the error messages. They often point to issues in your `.tf` files (e.g., syntax errors, incorrect resource names, missing required variables) or AWS permissions problems.
*   **Permissions Issues:** If you encounter `AccessDenied` errors, ensure the AWS credentials used by Terraform have the necessary IAM permissions to create/manage the resources defined.
*   **`your_ip` incorrect / SSH timeout:** If you can't SSH into the EC2 instance, double-check that the `your_ip` variable in `variables.tf` (or as provided during apply) is correct and that the security group `ec2_sg` correctly allows SSH (port 22) from this IP.

This completes the initial infrastructure setup. You can now proceed with installing and configuring the MLOps tools on the EC2 instance as outlined in the main project plan. 