import argparse
import threading

from utils.preemption import check_gcp_preemption


if __name__ == '__main__':
    preemption_event = threading.Event()

    # preemption_thread = threading.Thread(target=check_gcp_preemption, args=(preemption_event,))
    # training_thread = threading.Thread(target=cifar_train, args=(preemption_event,))

    # preemption_thread.start()
    # training_thread.start()

    # preemption_thread.join()
    # training_thread.join()

    parser = argparse.ArgumentParser(description='Train or test different ML models')
    parser.add_argument('--model', type=str, choices=['cifar', 'imagenet', 'bert'], help='Model to train or test')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode to run the model in')

    args = parser.parse_args()
    model = args.model
    mode = args.mode

    if model is None or mode is None:
        print('Model and mode (train/test) must be provided')

    if model == 'cifar':
        from models.cifar import train as cifar_train, test as cifar_test
        if mode == 'train':
            cifar_train(preemption_event=preemption_event)
        else:
            cifar_test()

    elif model == 'imagenet':
        from models.imagenet import train as imagenet_train, test as imagenet_test
        if mode == 'train':
            imagenet_train()
        else:
            imagenet_test()

    elif model == 'bert':
        from models.bert import train as bert_train, test as bert_test
        if mode == 'train':
            bert_train()
        else:
            bert_test()