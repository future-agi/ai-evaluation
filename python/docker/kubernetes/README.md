# Kubernetes Backend Setup

The Kubernetes backend runs each evaluation task as a K8s Job. Unlike
Celery/Temporal/Ray, there's no Docker Compose file — you need an actual
Kubernetes cluster.

## Quick Start (local with kind)

```bash
# 1. Create a local cluster
kind create cluster --name fi-evals

# 2. Build the eval runner image and load it into kind
docker build -t fi-eval-runner:latest .
kind load docker-image fi-eval-runner:latest --name fi-evals

# 3. Verify the image is available
kubectl get nodes

# 4. Use from Python
python -c "
from fi.evals.framework.backends import KubernetesBackend, KubernetesConfig
backend = KubernetesBackend(KubernetesConfig(image='fi-eval-runner:latest'))
print('Backend ready:', backend.name)
"
```

## With minikube

```bash
# 1. Start minikube
minikube start

# 2. Point docker to minikube's daemon
eval $(minikube docker-env)

# 3. Build inside minikube
docker build -t fi-eval-runner:latest .

# 4. Use with imagePullPolicy: Never
# KubernetesConfig(image="fi-eval-runner:latest", image_pull_policy="Never")
```

## Production

Push the image to your registry and configure accordingly:

```bash
docker build -t your-registry.com/fi-eval-runner:latest .
docker push your-registry.com/fi-eval-runner:latest
```

```python
backend = KubernetesBackend(KubernetesConfig(
    image="your-registry.com/fi-eval-runner:latest",
    namespace="evaluations",
))
```
