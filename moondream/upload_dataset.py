import sys
import os
import dotenv
dotenv.load_dotenv()

access_token = os.getenv("HF_ACCESS_TOKEN")
if not access_token:
    print("Please provide huggingface access token via HF_ACCESS_TOKEN.")
    sys.exit(1)

from huggingface_hub import HfApi
from huggingface_hub.constants import REPO_TYPE_DATASET

api = HfApi(token=access_token)

api.upload_large_folder(
    repo_id="athenlab/reva",
    folder_path="./data",
    repo_type=REPO_TYPE_DATASET,
    private=False,
    print_report=True,
)
