from huggingface_hub import HfApi
import os

HF_USERNAME = "keerthas"   # e.g. "sarassu"
REPO_NAME   = "tourism-dataset"    # repo id you want
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
HF_TOKEN    = "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"

# api = HfApi(token=os.getenv("HF_TOKEN"))
api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=REPO_ID,          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
