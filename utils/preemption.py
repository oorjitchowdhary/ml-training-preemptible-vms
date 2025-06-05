import requests, time, os
import logging

def check_gcp_preemption(preemption_event):
    while not preemption_event.is_set():
        try:
            url = 'http://metadata.google.internal/computeMetadata/v1/instance/preempted'
            headers = {'Metadata-Flavor': 'Google'}

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                preempted = response.text

                if preempted == 'TRUE':
                    logging.info('PREEMPTION: Preempted on GCP')
                    preemption_event.set()
                    return
                else:
                    logging.info('PREEMPTION: Not preempted on GCP')

            else:
                logging.error(f'PREEMPTION: Failed to check preemption status, status code: {response.status_code}')

        except Exception as e:
            logging.error(f'PREEMPTION: Error checking preemption status: {e}')

        time.sleep(5)


def check_simulated_preemption(preemption_event):
    while not preemption_event.is_set():
        try:
            if os.path.exists('/tmp/ml-training-preemptible-vms/preempted.txt'):
                logging.info('PREEMPTION: Preempted on simulated environment')
                preemption_event.set()

                if preemption_event.is_set():
                    logging.info('PREEMPTION: Preemption event set, stopping training')
                return
            else:
                logging.info('PREEMPTION: Not preempted in simulated environment')

        except Exception as e:
            logging.error(f'PREEMPTION: Error checking preemption status in simulated environment: {e}')

        finally:
            time.sleep(1)
