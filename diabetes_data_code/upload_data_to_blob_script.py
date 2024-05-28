import glob
import os

from azure.storage.blob import BlobClient, BlobServiceClient

from diabetes_data_code.config import load_env_vars

load_env_vars("config.yaml", "config_secret.yaml")
all_blobs = []
train_test_blobs = list(glob.glob(os.path.join("data", "training", "*.pkl")))
cv_blobs = list(glob.glob(os.path.join("data", "training", "cv", "*.pkl")))
all_blobs.extend(train_test_blobs)
all_blobs.extend(cv_blobs)

blob_service_client = BlobServiceClient.from_connection_string(
    os.environ["BLOB_CONNECTION_STRING"]
)
container_client = blob_service_client.get_container_client(
    os.environ["BLOB_CONTAINER_NAME"]
)

for blob_filename in all_blobs:
    with open(blob_filename, "rb") as f:
        blob_client = BlobClient.from_connection_string(
            conn_str=os.environ["BLOB_CONNECTION_STRING"],
            container_name=os.environ["BLOB_CONTAINER_NAME"],
            blob_name=blob_filename,
            max_block_size=1024 * 1024 * 4,
            max_single_put_size=1024 * 1024 * 8,
        )
        blob_client.upload_blob(f, overwrite=True, blob_type="BlockBlob")
