# from utils.preemption import is_preempted_on_gcp, is_simulated_preemption
from models.cifar import train as cifar_train, test as cifar_test
from models.imagenet import train as imagenet_train, test as imagenet_test

from utils.preemption import check_gcp_preemption

import threading

if __name__ == '__main__':
    preemption_event = threading.Event()

    preemption_thread = threading.Thread(target=check_gcp_preemption, args=(preemption_event,))
    training_thread = threading.Thread(target=cifar_train, args=(preemption_event,))

    preemption_thread.start()
    training_thread.start()

    preemption_thread.join()
    training_thread.join()
