
import subprocess
import mlflow
import os

import os

if os.getenv("GITHUB_ACTIONS") != "true":
    try:
        from pyngrok import ngrok
        ngrok_token = os.getenv("Ngrok")

        # Set your auth token here (replace with your actual token)
        ngrok.set_auth_token(ngrok_token)
        
        # Start MLflow UI on port 5000
        process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])
        
        # Create public tunnel
        public_url = ngrok.connect(5000).public_url
        print("MLflow UI is available at:", public_url)
    except ImportError:
        ngrok = None
else:
    ngrok = None

import os
access_token = os.getenv("Login")

# access_token = HfApi(token=os.environ["Login"])

# === USER EDITS ===
HF_USERNAME = "keerthas"                # e.g. "sarassu"
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-dataset"      # repo that contains Xtrain.csv etc
PATH_PREFIX_IN_DATASET = "split"                # "split" if files are at split/Xtrain.csv else "" if root
HF_TOKEN = access_token                          # paste token inside Colab only (must have read & write rights)
MODEL_REPO_NAME = "tourism-package-model"       # model repo to create/upload to: <HF_USERNAME>/<MODEL_REPO_NAME>
MLFLOW_TRACKING_URI = "http://localhost:8080"   # as you requested
EXPERIMENT_NAME = "Tourism-Package-Prediction-Experiment"
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Grid search hyperparams (example for RandomForest)
PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}
SCORING = "accuracy"
CV = 3
# ==================


# ---- imports ----
import os, json, tempfile, shutil, joblib
import pandas as pd, numpy as np
from huggingface_hub import hf_hub_download, HfApi, login, create_repo, upload_file, upload_folder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow, mlflow.sklearn

# ---- MLflow setup ----
# mlflow.set_tracking_uri(public_url)
mlflow.set_experiment(EXPERIMENT_NAME)
print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow experiment:", EXPERIMENT_NAME)

# ---- HF login ----
if not HF_TOKEN:
    raise SystemExit("Set HF_TOKEN variable with your Hugging Face token before running.")
login(token=HF_TOKEN)
api = HfApi()

# ---- helper to build repo-relative path ----
def repo_path(fname):
    return f"{PATH_PREFIX_IN_DATASET}/{fname}" if PATH_PREFIX_IN_DATASET else fname

# ---- download CSVs from HF dataset repo ----
files_to_fetch = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
local_files = {}
for f in files_to_fetch:
    path_in_repo = repo_path(f)
    print("Attempting to download:", path_in_repo)
    try:
        local = hf_hub_download(repo_id=DATASET_REPO_ID, filename=path_in_repo, repo_type="dataset", token=HF_TOKEN)
        print(" -> downloaded to", local)
        local_files[f] = local
    except Exception as e:
        raise SystemExit(f"Failed to download {path_in_repo} from {DATASET_REPO_ID}. Error: {e}\nCheck REPO_ID, path and HF_TOKEN and run the diagnostic repo listing cell if needed.")

# ---- load CSVs into pandas ----
X_train = pd.read_csv(local_files["Xtrain.csv"])
X_test  = pd.read_csv(local_files["Xtest.csv"])
y_train = pd.read_csv(local_files["ytrain.csv"])
y_test  = pd.read_csv(local_files["ytest.csv"])
# Flatten y if single column
if y_train.shape[1] == 1:
    y_train = y_train.iloc[:,0]
if y_test.shape[1] == 1:
    y_test = y_test.iloc[:,0]

print("Shapes: X_train", X_train.shape, "X_test", X_test.shape, "y_train", y_train.shape, "y_test", y_test.shape)

# ---- Separate numerical and categorical columns ----
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object","category","bool"]).columns.tolist()
# Move small-cardinality numeric to categorical if desired
for c in num_cols[:]:
    if X_train[c].nunique(dropna=True) <= 10:
        num_cols.remove(c)
        cat_cols.append(c)

print("Numerical cols:", num_cols)
print("Categorical cols:", cat_cols)

# ---- Preprocessing using make_column_transformer ----
num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])
preprocessor = make_column_transformer(
    (num_transformer, num_cols),
    (cat_transformer, cat_cols),
    remainder="drop"   # drop any other columns
)

# ---- Initialize base model (RandomForest here; change to XGB or others if you prefer) ----
base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

# ---- Build pipeline: preprocessing + model ----
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", base_model)
])

# ---- GridSearchCV initialization ----
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=PARAM_GRID,
    cv=CV,
    scoring=SCORING,
    n_jobs=-1,
    verbose=2
)

# ---- Start MLflow run & train ----
with mlflow.start_run(run_name="GridSearch_Tourism") as run:
    print("MLflow run id:", run.info.run_id)
    # Log grid settings
    mlflow.log_param("grid_param_keys", list(PARAM_GRID.keys()))
    mlflow.log_param("scoring", SCORING)
    mlflow.log_param("cv", CV)

    # Fit
    print("Fitting GridSearchCV — this may take a while depending on param grid")
    grid.fit(X_train, y_train)

    # Best params & estimator
    best_params = grid.best_params_
    best_estimator = grid.best_estimator_
    mlflow.log_param("best_params", json.dumps(best_params))
    print("Best params:", best_params)

    # Predictions
    y_train_pred = best_estimator.predict(X_train)
    y_test_pred  = best_estimator.predict(X_test)

    # Evaluate
    def compute_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        }

    train_metrics = compute_metrics(y_train, y_train_pred)
    test_metrics  = compute_metrics(y_test, y_test_pred)

    # Log metrics
    for k,v in train_metrics.items():
        mlflow.log_metric(f"train_{k}", v)
    for k,v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    # Save model locally with joblib and log as artifact
    import os
    from pathlib import Path
    
    BASE = os.getenv("OUTPUT_BASE")
    out_dir = Path(BASE) / "best_model_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # out_dir = "/content/best_model_pipeline"
    # os.makedirs(out_dir, exist_ok=True)
    model_path = out_dir / "best_pipeline.joblib"
    joblib.dump(best_estimator, model_path)
    mlflow.log_artifact(model_path.as_posix(), artifact_path="model_artifact")

    # Save params & metrics JSON
    meta = {
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }

    meta_path = out_dir / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
mlflow.log_artifact(meta_path.as_posix(), artifact_path="model_artifact")

print("Logged model and metadata to MLflow artifacts at:", out_dir)

    # with open(os.path.join(out_dir, "metadata.json"), "w") as f:
    #     json.dump(meta, f, indent=2)
    # mlflow.log_artifact(os.path.join(out_dir, "metadata.json"), artifact_path="model_artifact")

    # print("Logged model and metadata to MLflow artifacts.")

# ---- After MLflow run: Register/upload the model to Hugging Face model hub ----
MODEL_REPO_ID = f"{HF_USERNAME}/{MODEL_REPO_NAME}"
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True)
print("Model repo ready:", MODEL_REPO_ID)

# Add a README/model card
model_card_text = f"""# {MODEL_REPO_NAME}

Model trained with GridSearchCV on dataset: {DATASET_REPO_ID}
Best params: {best_params}
Train metrics: {train_metrics}
Test metrics: {test_metrics}

Saved artifact: best_pipeline.joblib (joblib.load to load)
"""
with open(os.path.join(out_dir, "README.md"), "w") as f:
    f.write(model_card_text)

# Upload folder content (model file, metadata, readme)
upload_folder(
    folder_path=out_dir,
    path_in_repo=".",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Upload trained pipeline, metadata, and README from Colab"
)
print("Uploaded model files to Hugging Face model hub:", f"https://huggingface.co/{MODEL_REPO_ID}")

# Also upload a small model card as README (already included above)
print("Done. Best model artifact and metadata are uploaded to the HF model hub.")
# ---------------------------------------------------------------------------------------------
