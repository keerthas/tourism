# hosting.py
from huggingface_hub import HfApi, create_repo
import os
from pathlib import Path
import sys

HF_USERNAME = "keerthas"   # e.g. "sarassu"
REPO_NAME   = "tourism-dataset"    # repo id you want
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"

# Prefer standard env var HF_TOKEN, but keep fallback to older "Login" if present
HF_TOKEN = os.getenv("Login")

if not HF_TOKEN:
    sys.exit("ERROR: HF token not found. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN / Login) in environment.")

api = HfApi(token=HF_TOKEN)

# --- Choose repo_type: "dataset" because REPO_NAME is a dataset.
# If you intended to upload to a Space, set repo_type to "space" and ensure the repo exists.
REPO_TYPE = "dataset"   # change to "space" only if you're uploading a Space and repo exists/created as a Space

# Ensure repo exists (create if it doesn't)
try:
    api.repo_info(repo_id=REPO_ID, repo_type=REPO_TYPE)
    print(f"Repo {REPO_ID} exists as {REPO_TYPE}.")
except Exception as e:
    print(f"Repo {REPO_ID} not found as {REPO_TYPE}. Creating it...")
    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
    print("Created repo:", REPO_ID)

# Local folder you want to upload — adjust if needed
local_folder = Path("tourism_project/deployment")
if not local_folder.exists():
    sys.exit(f"ERROR: local folder to upload not found: {local_folder.resolve()}")

# Upload folder (non-interactive)
try:
    api.upload_folder(
        folder_path=str(local_folder),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        path_in_repo="",           # optional: subfolder in repo
        commit_message="Upload deployment files from CI"
    )
    print("Upload complete.")
except Exception as e:
    raise RuntimeError(f"Failed to upload folder to Hugging Face: {e}")
