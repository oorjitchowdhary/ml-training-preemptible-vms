import requests

def is_preempted_on_gcp():
    METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/preempted'
    METADATA_HEADERS = {'Metadata-Flavor': 'Google'}

    try:
        response = requests.get(METADATA_URL, headers=METADATA_HEADERS)
        if response.status_code == 200:
            return response.text.strip() == 'TRUE'
    except Exception as e:
        print(f'Error detecting preemption: {e}')

    return False

def is_simulated_preemption():
    try:
        with open('./preemption.txt', 'r') as f:
            return f.read().strip() == 'TRUE'
    except Exception as e:
        print(f'Error reading preemption file: {e}')
    
    return False
