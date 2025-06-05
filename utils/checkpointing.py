from google.cloud import storage
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
service_account_json = 'robs-project-382021-29095b54cf4c.json' # Replace with your own service account json
service_account_json_path = os.path.join(dir_path, '../', service_account_json)

bucket_name = 'cifar-pytorch-checkpoints' # Replace with your own bucket name
storage_client = storage.Client().from_service_account_json(service_account_json_path)

def save_checkpoint_to_gcp(filename):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_filename(f'./checkpoints/{filename}')
        print(f'Checkpoint saved to GCP: {filename}')
        return True
    except Exception as e:
        print(f'Error saving checkpoint to GCP: {e}')
        return False

def load_checkpoint_from_gcp(filename):
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(filename)
    os.makedirs('./checkpoints', exist_ok=True)
    blob.download_to_filename(f'./checkpoints/{filename}')

    print(f'Checkpoint loaded from GCP: {filename}')

def checkpoint_exists(filename):
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(filename)
    return blob.exists()

def resume_from_checkpoint():
    # Find the latest checkpoint
    epoch, latest_checkpoint = 0, None
    for i in range(9, -1, -1):
        if checkpoint_exists(f'checkpoint_{i}.pth'):
            latest_checkpoint = f'checkpoint_{i}.pth'
            epoch = i
            break
    
    if latest_checkpoint is None:
        print('No checkpoints found')
        return 0, None
    
    # Load the latest checkpoint
    load_checkpoint_from_gcp(latest_checkpoint)
    return epoch + 1, latest_checkpoint