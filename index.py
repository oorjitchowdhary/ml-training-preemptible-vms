import argparse
import threading
import os

from utils.preemption import check_gcp_preemption, check_simulated_preemption
from models.cifar import train as cifar_train, test as cifar_test

if __name__ == '__main__':
    preemption_event = threading.Event()

    preemption_thread = threading.Thread(target=check_simulated_preemption, args=(preemption_event,))
    training_thread = threading.Thread(target=cifar_train, args=(preemption_event,))

    preemption_thread.start()
    training_thread.start()

    preemption_thread.join()
    training_thread.join()

    if not preemption_event.is_set():
        print('Training completed successfully')

    # parser = argparse.ArgumentParser(description='Premptively checkpoint different ML tasks')
    # parser.add_argument('--datatype', type=str, choices=['text', 'image'], help='Type of the task to perform')
    # parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'bert'], help='Dataset to use')
    # parser.add_argument('--provider', type=str, choices=['gcp', 'aws', 'azure', 'hyak'], help='Compute provider to run the model on')
    # parser.add_argument('--task', type=str, choices=['training', 'testing', 'inference'], help='Task to perform')

    # args = parser.parse_args()
    # datatype = args.datatype
    # dataset = args.dataset
    # provider = args.provider
    # task = args.task

    # if not datatype or not dataset or not provider or not task:
    #     raise ValueError('Please provide all the necessary arguments')

    # if dataset == 'cifar':
    #     from models.cifar import train as cifar_train, test as cifar_test
    #     if task == 'training':
    #         cifar_train(preemption_event=preemption_event)
    #     else:
    #         cifar_test()

    # elif dataset == 'imagenet':
    #     from models.imagenet import train as imagenet_train, test as imagenet_test
    #     if task == 'training':
    #         imagenet_train()
    #     else:
    #         imagenet_test()

    # elif dataset == 'bert':
    #     from models.bert import train as bert_train, test as bert_test
    #     if task == 'training':
    #         bert_train()
    #     else:
    #         bert_test()
