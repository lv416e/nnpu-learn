resource "aws_s3_bucket" "sagemaker_bucket" {
  bucket        = var.bucket_name
  force_destroy = true
}