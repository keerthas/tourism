# app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download

# Prefer HF_TOKEN env var; fallback to Login only if present
HF_TOKEN = os.getenv("Login")
if not HF_TOKEN:
    raise RuntimeError("ERROR: HF token not found in environment. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN / Login).")

# IMPORTANT: hf_hub_download expects repo_id as "username/repo" (NOT a full URL)
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "keerthas/tourism-package-model")  # set via env if needed
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_pipeline.joblib")

app = FastAPI(title="Tourism Package Prediction Service")

def load_model_from_hf():
    try:
        # If the model repo is actually a model repo, repo_type defaults to "model" so no need to pass repo_type.
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN)
        model = joblib.load(model_path)
        print("Model loaded from:", model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to download/load model from HF ({HF_MODEL_REPO}/{MODEL_FILENAME}): {e}")

MODEL = load_model_from_hf()

class Record(BaseModel):
    __root__: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(records: Record):
    try:
        df = pd.DataFrame(records.__root__)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    try:
        preds = MODEL.predict(df)
        proba = MODEL.predict_proba(df).tolist() if hasattr(MODEL, "predict_proba") else None
        return {"predictions": preds.tolist(), "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")
