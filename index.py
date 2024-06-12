# from utils.preemption import is_preempted_on_gcp, is_simulated_preemption
from models.cifar import train as cifar_train, test as cifar_test
from models.imagenet import train as imagenet_train, test as imagenet_test

if __name__ == '__main__':
    cifar_train()
    cifar_test()
