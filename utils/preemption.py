import requests, time


def check_gcp_preemption(preemption_event):
    while not preemption_event.is_set():
        try:
            url = 'http://metadata.google.internal/computeMetadata/v1/instance/preempted'
            headers = {'Metadata-Flavor': 'Google'}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                preempted = response.text

                if preempted == 'TRUE':
                    print('Preempted on GCP')
                    preemption_event.set()
                    return
                else:
                    print('Not preempted on GCP')

            else:
                print('Failed to check GCP preemption with HTTP status code:', response.status_code)

        except Exception as e:
            print(e)

        time.sleep(1)