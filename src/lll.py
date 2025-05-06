`   ``python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

# Define the tasks
tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

# Define the transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Download the MNIST dataset
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create the data loaders for each task
train_loaders = []
test_loaders = []

for task in tasks:
    train_dataset = []
    test_dataset = []
    for i in range(len(mnist_train)):
        if mnist_train.targets[i] in task:
            train_dataset.append(mnist_train[i])
    for i in range(len(mnist_test)):
        if mnist_test.targets[i] in task:
            test_dataset.append(mnist_test[i])
            
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

# Initialize the model and optimizer
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model on each task sequentially
for i, task in enumerate(tasks):
    print(f"Training on task {i+1}: {task}")
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loaders[i]):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model on the current task
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loaders[i]:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy on task {i+1}: {100 * correct / total:.2f}%")
```
