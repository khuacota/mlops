"""API de inferencia: carga el modelo exportado y responde predicciones."""
import json
import os
import socket
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))

model = mlflow.sklearn.load_model(str(MODEL_DIR / "artifacts"))
model_info = json.loads((MODEL_DIR / "model_info.json").read_text())

app = FastAPI(title="Predicción de pedidos atrasados")


class Order(BaseModel):
    distance_km: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    num_items: float = Field(..., ge=1)
    dispatch_hour: float = Field(..., ge=0, le=23)
    warehouse_load: float = Field(..., ge=0, le=1)
    weather_risk: float = Field(..., ge=0, le=1)
    carrier: Literal["A", "B", "C"]
    shipping_type: Literal["standard", "express"]


@app.get("/health")
def health():
    return {"status": "ok", "pod": socket.gethostname()}


@app.get("/info")
def info():
    return {**model_info, "pod": socket.gethostname()}


@app.post("/predict")
def predict(order: Order):
    df = pd.DataFrame([order.model_dump()])
    proba = float(model.predict_proba(df)[0, 1])
    return {
        "late": int(proba >= 0.5),
        "probability_late": round(proba, 4),
        "model_version": model_info["version"],
        "run_id": model_info["run_id"],
        "pod": socket.gethostname(),
    }