import os, torch, torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import threading

from utils.checkpointing import save_checkpoint_to_gcp, resume_from_checkpoint

# Define the CNN model for CIFAR-10 dataset
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Lock for checkpointing
checkpoint_lock = threading.Lock()

# Train the CNN model on CIFAR-10 dataset
def train(preemption_event):
    while not preemption_event.is_set():
        # Define image transformations
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Number of images to load per batch
        batch_size = 4

        # Load CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Classes in CIFAR-10 dataset for reference
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Get CNN model and define loss function and optimizer
        net = CifarNet()

        import torch.optim as optim
        import torch.nn as nn

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train the network
        start_epoch, latest_checkpoint = resume_from_checkpoint()
        for epoch in range(start_epoch, 10):
            if preemption_event.is_set():
                print('Training at epoch', epoch, 'stopped due to preemption')
                break

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                if preemption_event.is_set():
                    print('Training loop stopped due to preemption')
                    break

                inputs, labels = data

                optimizer.zero_grad()

                if latest_checkpoint:
                    checkpoint_data = torch.load(f'./checkpoints/{latest_checkpoint}')
                    net.load_state_dict(checkpoint_data['model_state_dict'])
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    print(f'Resuming from checkpoint: {latest_checkpoint}')

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
            if not preemption_event.is_set():
                with checkpoint_lock:
                    if not preemption_event.is_set():
                        os.makedirs('./checkpoints', exist_ok=True)
                        torch.save(checkpoint, f'./checkpoints/checkpoint_{epoch}.pth')
                        save_checkpoint_to_gcp(f'checkpoint_{epoch}.pth')
                        print(f'Checkpoint saved for epoch {epoch}')


# Test the CNN model on CIFAR-10 dataset
def test():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = CifarNet()
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