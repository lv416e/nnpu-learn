#-------------------------------------------------------
# Provider Configurations & Requirements
#-------------------------------------------------------
terraform {
  required_version = "~> 1.5.1"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.67.0"
    }
  }
}

provider "aws" {
  region  = var.AWS_REGION
  profile = var.AWS_PROFILE
}

module "bucket" {
  source      = "./modules/bucket"
  bucket_name = var.BUCKET_NAME
}

module "permissions" {
  source                   = "./modules/permissions"
  sagemaker_exec_role_name = var.SAGEMAKER_EXEC_ROLE_NAME
}

module "repository" {
  source                    = "./modules/repository"
  sagemaker_repository_name = var.SAGEMAKER_REPOSITORY_NAME
}