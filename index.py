import os, sys, time
import torch, torchvision
import torchvision.transforms as transforms

# from checkpointing import save_checkpoint_to_gcp, resume_from_checkpoint
from preemption import is_preempted_on_gcp, is_simulated_preemption
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
    # start_epoch, latest_checkpoint = resume_from_checkpoint()
    start_epoch = 0
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
        # save_checkpoint_to_gcp(f'checkpoint_{epoch}.pth')

    # Save the trained model
    torch.save(net.state_dict(), f'./checkpoints/final_model.pth')
    # save_checkpoint_to_gcp('final_model.pth')
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

def imagenet():
    transform_train = transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = torchvision.datasets.ImageNet(root='./images', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.ImageNet(root='./images', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Use pre-trained ResNet-50 model
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 1000)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the network
    for epoch in range(90):
        resnet50.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{90}, Loss: {epoch_loss}')

        scheduler.step()

    # Save the trained model
    torch.save(resnet50.state_dict(), 'resnet50.pth')

    # Test the network
    resnet50.eval()
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}')


if __name__ == '__main__':
    # # while True:
    # #     if is_simulated_preemption(): # for testing based on file presence
    # #         print('Preempted; performing graceful shutdown...')
    # #         # save_checkpoint_to_gcp('checkpoint.pth')
    # #         sys.exit(0)
    # #     else:
    #     train()
    #     print('Training complete; moving to testing...')
    #     test()
    #     print('Testing complete; all done!')

    #     # time.sleep(10)

    imagenet()