resource "aws_ecr_repository" "sagemaker_images" {
  name                 = var.sagemaker_repository_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}