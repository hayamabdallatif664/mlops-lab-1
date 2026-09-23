import io
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.environ.get("MODEL_URI", "models:/food11@champion")

# Same order as torchvision's ImageFolder assigned during training (sorted folder names)
CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

# Must match the evaluation transform used in train.py
IMAGE_SIZE = (128, 128)
preprocess = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = None


@asynccontextmanager
async def lifespan(app):
    # Load the model once at startup, resolved through the Model Registry alias
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Loading {MODEL_URI} from {MLFLOW_TRACKING_URI}")
    model = mlflow.pyfunc.load_model(MODEL_URI)
    print("Model loaded")
    yield


app = FastAPI(title="Food-11 classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    batch = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
    logits = np.asarray(model.predict(batch))[0]

    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    best = int(probs.argmax())

    return {"category": CLASSES[best], "confidence": round(float(probs[best]), 4)}
