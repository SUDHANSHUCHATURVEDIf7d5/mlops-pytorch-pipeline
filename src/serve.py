import io
import os
import yaml
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from model import get_model
import uvicorn

app = FastAPI(title="CIFAR-10 ML Model API")

# Try to load config if it exists, otherwise use defaults
config_path = os.getenv("CONFIG_PATH", "configs/training_config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]
    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"], config["output"]["model_name"])
else:
    architecture = os.getenv("MODEL_ARCHITECTURE", "resnet18")
    num_classes = int(os.getenv("NUM_CLASSES", "10"))
    checkpoint_path = os.getenv(
        "MODEL_PATH", "/app/checkpoints/classifier_v1.pt")

device = torch.device("cpu")

# Initialize model
try:
    model = get_model(architecture=architecture, num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model_loaded = True
        print(f"Successfully loaded model checkpoint from {checkpoint_path}")
    else:
        model_loaded = False
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Model is not loaded properly.")
except Exception as e:
    model_loaded = False
    print(f"Error initializing model: {e}")

# Preprocessing transforms for CIFAR-10
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


@app.get("/health")
async def health():
    if model_loaded:
        return JSONResponse(status_code=200, content={"status": "healthy"})
    else:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": "Model not loaded"})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    try:
        # Read image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Apply transforms and add batch dimension
        tensor = transform(pil_image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Convert to list
        probs_list = probabilities.squeeze().tolist()
        return JSONResponse(status_code=200, content={"probabilities": probs_list})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
