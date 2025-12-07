
import os
access_key = os.getenv("Login")

# access_key = HfApi(token=os.getenv("Login"))

# access_key = os.getenv("Login")

# === USER EDITS ===
HF_USERNAME = "keerthas"         # e.g. "sarassu"
REPO_NAME   = "tourism-dataset"                  # dataset repo name (will use <username>/tourism)
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
HF_TOKEN    = access_key                # your Hugging Face token (must have write rights)
# https://huggingface.co/datasets/keerthas/tourism-dataset
# https://huggingface.co/datasets/keerthas/tourism-dataset/tree/main/data
# DATASET_PATH = f"hf://datasets/{HF_USERNAME}/{REPO_NAME}/tourism.csv"
DATASET_PATH = f"hf://datasets/{HF_USERNAME}/{REPO_NAME}"
FILENAME_IN_REPO = "data/tourism.csv"         # path inside the repo (relative)
TARGET_COLUMN = None                     # optional: set to your target column name (e.g. "ProdTaken"). If None, we'll try to auto-detect.
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ==================

from huggingface_hub import hf_hub_download, HfApi, login, create_repo, upload_file
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Login to HF programmatically (so uploads work)
if not HF_TOKEN:
    raise SystemExit("Please set HF_TOKEN variable with your Hugging Face token before running.")
login(token=HF_TOKEN)

# Ensure dataset repo exists (create if missing)
try:
    create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    print("Dataset repo ready:", REPO_ID)
except Exception as e:
    print("Warning while creating/checking repo:", e)



# ---- Step 4: Download/read the CSV file from the dataset repo ----
# Two ways: (1) attempt hf_hub_download (works if file present in repo), (2) fallback to datasets.load_dataset
local_download_path = None
try:
    # hf_hub_download expects repo_id and filename relative to repo root
    local_download_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_IN_REPO, repo_type="dataset", token=HF_TOKEN)
    print("Downloaded file from HF to:", local_download_path)
except Exception as e:
    print("hf_hub_download failed:", e)
    print("local_download_path",local_download_path)
    # try loading with datasets (useful if dataset has a 'train' split)
    try:
        from datasets import load_dataset
        ds = load_dataset(REPO_ID, split="train", use_auth_token=HF_TOKEN)
        df = ds.to_pandas()
        local_download_path = "/content/_hf_fallback_loaded.csv"
        df.to_csv(local_download_path, index=False)
        print("Loaded dataset via datasets.load_dataset fallback. Saved to:", local_download_path)
    except Exception as e2:
        raise SystemExit("Could not fetch CSV from HF. Check REPO_ID/FILENAME_IN_REPO and token. Error2: " + str(e2))

# Read CSV into pandas
read_success = False
for enc in ("utf-8", "latin1", "iso-8859-1"):
    try:
        df = pd.read_csv(local_download_path, encoding=enc, low_memory=False)
        read_success = True
        break
    except Exception:
        pass
if not read_success:
    raise SystemExit("Failed to read downloaded CSV. Check file content/encoding.")

print("Original dataframe shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---- Step 5: Separate target and independent variables ----
# If user provided TARGET_COLUMN use that; else try to auto-detect common names.
if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    target_col = TARGET_COLUMN
else:
    # Common heuristics for auto detection
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["target","label","class","prod","taken","outcome","y"])]
    if candidates:
        target_col = candidates[0]
    else:
        # fallback: if any column has small cardinality 2..20 and looks like label choose it
        small_card = [c for c in df.columns if 2 <= df[c].nunique(dropna=True) <= 20]
        target_col = small_card[0] if small_card else None

if target_col is None:
    raise SystemExit("Could not detect target column automatically. Set TARGET_COLUMN variable manually.")
print("Using target column:", target_col)

X = df.drop(columns=[target_col]).copy()
y = df[[target_col]].copy()

# ---- Step 6: Separate categorical and numerical variables of X ----
# We'll treat object and category dtypes as categorical; numbers as numerical.
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Additionally, columns with small unique counts but numeric-type may be categorical
for c in num_cols[:]:
    if X[c].nunique(dropna=True) <= 10:
        # move to categorical
        num_cols.remove(c)
        cat_cols.append(c)

print("Numerical columns:", num_cols)
print("Categorical columns:", cat_cols)

# ---- Step 7: Split into train/test sets (stratify if possible) ----
# If y has only single column use its values for stratify
stratify_col = None
if y.shape[1] == 1:
    # ensure no NaNs for stratify
    if y[target_col].isna().sum() == 0 and y[target_col].nunique() > 1:
        stratify_col = y[target_col]
if stratify_col is not None:
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_col)
    print("Performed stratified split on target.")
else:
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print("Performed random split (no stratify).")

print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain.shape, "ytest:", ytest.shape)

# # ---- Step 8: Save train/test CSVs locally ----
# out_dir = "/content/processed"
# os.makedirs(out_dir, exist_ok=True)

# ---- Step 8: Save train/test CSVs locally ----
import os
from pathlib import Path

BASE = os.getenv("OUTPUT_BASE")
out_dir = Path(BASE) / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Writing outputs to: {out_dir}")

Xtrain_path = os.path.join(out_dir, "Xtrain.csv")
Xtest_path  = os.path.join(out_dir, "Xtest.csv")
ytrain_path = os.path.join(out_dir, "ytrain.csv")
ytest_path  = os.path.join(out_dir, "ytest.csv")

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Saved files locally in", out_dir)
print(" -", Xtrain_path)
print(" -", Xtest_path)
print(" -", ytrain_path)
print(" -", ytest_path)

# ---- Step 9: Upload these CSVs into Hugging Face dataset repo ----
api = HfApi()

# Make a subfolder path in the repo (optional). We'll put them under 'split/' folder
path_in_repo_prefix = "split"

for local_file in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    fname = os.path.basename(local_file)
    path_in_repo = f"{path_in_repo_prefix}/{fname}"
    try:
        upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            commit_message=f"Upload {fname} from Colab"
        )
        print("Uploaded to HF:", path_in_repo)
    except Exception as e:
        print("Failed to upload", fname, "Error:", e)

print("All done. Visit your dataset at: https://huggingface.co/datasets/" + REPO_ID)
