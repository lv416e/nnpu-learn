# Implementation of Positive Unlabeled Learning

## Requirements:

| Dependencies | Version |
|--------------|---------|
| Python       | 3.10.4+ |
| numpy        | 1.22.3+ |
| torch        | 1.11.0+ |
| torchvision  | 0.12.0+ |
| awscli       | 2.12.1+ |
| terraform    | 1.5.1+  |

## Usage

`cd` to the root of this project directory

Start Containers:

```docker
docker compose -f docker/docker-compose.yml up engine
```

Stop Containers:

```docker
docker compose -f docker/docker-compose.yml down
```

## Reference

[1] Ryuichi Kiryo, Gang Niu, Marthinus Christoffel du Plessis, and Masashi Sugiyama. (2017).  
_Positive-Unlabeled Learning with Non-Negative Risk Estimator._,  
Advances in neural information processing systems.  
https://arxiv.org/pdf/1703.00593.pdf