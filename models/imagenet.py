import torch, torchvision
import torchvision.transforms as transforms

# Use pre-trained ResNet-50 model
resnet50 = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, 1000)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50.to(device)

def train():
    transform_train = transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     ])

    train_dataset = torchvision.datasets.ImageNet(root='./images', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

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

def test():
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_dataset = torchvision.datasets.ImageNet(root='./images', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    resnet50.eval()
    criterion = torch.nn.CrossEntropyLoss()
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