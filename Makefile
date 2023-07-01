include .env

ECR_REPOSITORY := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
DOCKER_IMG := $(REPOSITORY_NAME)
DOCKER_TAG := $(VERSION)

.PHONY: all
all:
	help

docker-build:
	docker compose -f docker/docker-compose.yaml --env-file .env build training-job

docker-push:
	aws ecr get-login-password --profile $(AWS_PROFILE) --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_REPOSITORY)
	docker tag $(DOCKER_IMG):$(DOCKER_TAG) $(ECR_REPOSITORY)/$(DOCKER_IMG):$(DOCKER_TAG) && docker tag $(DOCKER_IMG):$(DOCKER_TAG) $(ECR_REPOSITORY)/$(DOCKER_IMG):latest
	docker push $(ECR_REPOSITORY)/$(DOCKER_IMG):$(DOCKER_TAG) && docker push $(ECR_REPOSITORY)/$(DOCKER_IMG):latest

tf-plan:
	cd terraform \
	&& terraform init \
	&& terraform fmt \
	&& terraform validate \
	&& terraform plan

tf-deploy:
	cd terraform \
	&& terraform fmt \
	&& terraform validate \
	&& terraform destroy -auto-approve \
	&& terraform apply -auto-approve

tf-apply:
	cd terraform \
	&& terraform init \
	&& terraform fmt \
	&& terraform validate \
	&& terraform apply -auto-approve

tf-destroy:
	cd terraform \
	&& terraform init \
	&& terraform destroy -auto-approve