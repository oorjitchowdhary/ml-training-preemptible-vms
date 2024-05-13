import os, sys
import torch, torchvision
import torchvision.transforms as transforms

from checkpointing import save_checkpoint_to_gcp, resume_from_checkpoint
from preemption import is_preempted_on_gcp
from model import Net

def train():
    # Define image transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Number of images to load per batch
    batch_size = 4

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Classes in CIFAR-10 dataset for reference
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get CNN model and define loss function and optimizer
    net = Net()

    import torch.optim as optim
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    start_epoch, latest_checkpoint = resume_from_checkpoint()
    for epoch in range(start_epoch, 10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch}, {i + 1}] loss: {running_loss / 2000}')
                running_loss = 0.0

        # Save the checkpoint at the end of each epoch
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(checkpoint, f'./checkpoints/checkpoint_{epoch}.pth')
        save_checkpoint_to_gcp(f'checkpoint_{epoch}.pth')

    # Save the trained model
    torch.save(net.state_dict(), f'./checkpoints/final_model.pth')
    save_checkpoint_to_gcp('final_model.pth')
    print('Final model saved')

def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()
    net.load_state_dict(torch.load('./checkpoints/final_model.pth'))

    # Test the network on the whole dataset
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10,000 test images: {100 * correct / total}%')

if __name__ == '__main__':
    while not is_preempted_on_gcp():
        train()
        print('Training complete; moving to testing...')
    else:
        print('Preempted; train() saved latest checkpoint; exiting...')
        # save_checkpoint_to_gcp('checkpoint.pth')
        sys.exit(0)
    # test()
    # print('Testing complete; all done!')