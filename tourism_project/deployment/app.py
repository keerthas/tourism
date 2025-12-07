import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download

from google.colab import userdata
access_token = userdata.get("Login") 

# HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "<YOUR_HF_USERNAME>/<YOUR_MODEL_REPO>")
# MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_pipeline.joblib")
# HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_MODEL_REPO = "https://huggingface.co/keerthas/tourism-package-model"
MODEL_FILENAME = "best_pipeline.joblib"
HF_TOKEN = access_token

app = FastAPI(title="Tourism Package Prediction Service")

def load_model_from_hf():
    try:
        path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN)
        model = joblib.load(path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to download/load model from HF: {e}")

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
