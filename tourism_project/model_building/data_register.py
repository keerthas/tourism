
# from google.colab import userdata
# access_key = userdata.get("Login") 

# access_key = HfApi(token=os.getenv("Login"))

# access_token = HfApi(token=os.environ["Login"])

# 2) Set variables - EDIT these!
HF_USERNAME = "keerthas"   # e.g. "sarassu"
REPO_NAME   = "tourism-dataset"    # repo id you want
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
HF_TOKEN    =  os.environ.get("Login", None)          # Paste your Hugging Face token here

# 3) Create the dataset repo (if it already exists this will raise - we'll handle below)
from huggingface_hub import HfApi, upload_folder, create_repo, login
api = HfApi()

# Login programmatically (so upload_folder can use the token)
login(token=HF_TOKEN)

# Create repo if not exists
try:
    create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    print("Repo created/exists:", REPO_ID)
except Exception as e:
    print("create_repo warning:", e)

# 4) Upload the local folder /content/data to the dataset repo root (or into a 'data/' subfolder)
local_folder = "tourism_project/data"   # the folder you mentioned
if not os.path.exists(local_folder):
    raise SystemExit("Local folder not found: " + local_folder)

# upload_folder will recursively upload files and create commits
upload_folder(
    folder_path=local_folder,
    path_in_repo="data",       # files will be uploaded under /data in the dataset repo
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Upload tourism data from Colab"
)
print("Upload finished. Visit: https://huggingface.co/datasets/" + REPO_ID)
