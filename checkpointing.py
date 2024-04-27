from google.cloud import storage
import io

bucket_name = 'cifar-pytorch-checkpoints'
storage_client = storage.Client().from_service_account_json('robs-project-382021-29095b54cf4c.json')

def save_checkpoint_to_gcp(filename):
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(filename)
    blob.upload_from_filename(f'./checkpoints/{filename}')

    print(f'Checkpoint saved to GCP: {filename}')

def load_checkpoint_from_gcp(filename):
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(filename)
    blob.download_to_filename(f'./gcp/{filename}')

    print(f'Checkpoint loaded from GCP: {filename}')

def checkpoint_exists(filename):
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(filename)
    return blob.exists()
