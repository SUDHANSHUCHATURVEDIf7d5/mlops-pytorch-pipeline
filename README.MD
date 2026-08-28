# MLOps PyTorch Pipeline

This repository contains a full deployment lifecycle for a PyTorch image classification model (ResNet-18 on CIFAR-10) using Docker and Kubernetes. It covers local development, containerized training, and orchestrating a highly available serving endpoint.

## Architecture Diagram

```mermaid
graph TD
    subgraph cluster ["Kubernetes Cluster (ml-training namespace)"]
        cfg[ConfigMap: training-config]
        pvc[(PersistentVolumeClaim: training-pvc)]
        
        job[Job: model-training]
        deploy[Deployment: model-serving]
        svc[Service: model-serving]
        hpa[HPA: model-serving-hpa]
        
        cfg -->|Mounts configs| job
        cfg -->|Mounts configs| deploy
        
        job -->|Trains & writes model| pvc
        pvc -->|Reads model (ReadOnly)| deploy
        
        deploy --- svc
        hpa -.->|Autoscales CPU| deploy
    end
    
    User((User)) -->|POST /predict| svc
```

## Setup Instructions

### Prerequisites
- Docker Desktop (with Kubernetes enabled) or Minikube
- `kubectl` CLI installed
- Python 3.10+

### 1. Build the Docker Images
From the root of the repository, build both the training and serving images using the provided multi-stage Dockerfiles:

```bash
# Build the training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Build the serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 2. Run the Kubernetes Training Job
Deploy the Kubernetes manifests to create the namespace, configmap, persistent volume, and start the training job.

```bash
# Apply the training manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# Monitor the training progress
kubectl get pods -n ml-training -w
kubectl logs -l job-name=model-training -n ml-training -f
```

*Note: You must wait for the `model-training` pod to reach the `Completed` state before moving to step 3, otherwise the serving pods will enter a `CrashLoopBackOff` while waiting for the model checkpoint to be saved.*

### 3. Deploy the Serving Endpoint
Once training is complete, deploy the FastAPI serving endpoint:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Wait for the serving pods to become `READY`:
```bash
kubectl get pods -n ml-training
```

### 4. Test the API
Port-forward the service to your local machine:
```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

In a separate terminal, test the endpoints:
```bash
# Check health
curl http://localhost:8080/health

# Run a prediction
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Expected Results

### Training Performance
After 10 epochs of training the ResNet-18 model on the CIFAR-10 dataset using CPU processing, you should expect to see the validation accuracy reach approximately **77% - 78%**.

Sample end-of-training log output:
```json
{"epoch": 10, "train_loss": 0.5119, "train_accuracy": 0.8268, "val_loss": 0.4903, "val_accuracy": 0.8347}
{"event": "checkpoint_saved", "path": "checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.4903}
```

### Prediction Output
When sending a sample image to the `/predict` endpoint, the FastAPI server will return the top-1 predicted class and its confidence score:

```json
{
  "predicted_class": "cat",
  "confidence": 0.9238197207450867
}
{"probabilities":[0.00031145731918513775,2.179776674893219e-05,0.05529380962252617,0.009759151376783848,0.010387813672423363,0.00011636040289886296,0.9238197207450867,0.00011934142094105482,5.00528585689608e-05,0.0001204947693622671]}%    
```
