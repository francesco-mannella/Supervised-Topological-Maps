import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np

from stm.topological_maps import TopologicalMap, LossFactory


class Efficacies:
    def __init__(self, shape, radial_sigma, decay):
        self._efficacies = torch.zeros(shape)
        self.radial_sigma = radial_sigma
        self.decay = decay

    def update(self, errors, radial_sigma):
        error_radial_bases = torch.exp(-0.5 * self.radial_sigma * errors**2)
        self._efficacies = self.efficacies + self.decay * (
            error_radial_bases - self._efficacies
        )

    def get_inefficacies(self):

        return 1 - self._efficacies


# Define the tasks
tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

# Define the transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    ]
)


# Define hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 2
radial_sigma = 1.0
decay = 0.2
input_dim = 784
latent_dim = 100
anchor_sigma = 2.0
neigh_sigma_max = 10
neigh_sigma_base = 0.6

anchors = torch.tensor(
    [
        [0.15, 0.17],
        [0.12, 0.54],
        [0.16, 0.84],
        [0.50, 0.15],
        [0.36, 0.45],
        [0.62, 0.50],
        [0.48, 0.82],
        [0.83, 0.17],
        [0.88, 0.50],
        [0.83, 0.83],
    ]
) * np.sqrt(latent_dim)

# Download the MNIST dataset
mnist_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    train_loaders.append(train_loader)
    test_loaders.append(test_loader)

# Initialize the model and optimizer
model = TopologicalMap(input_dim, latent_dim)
lossManager = LossFactory(model, "stm")
efficacies = Efficacies(
    (batch_size, latent_dim),
    radial_sigma=radial_sigma,
    decay=decay,
)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model on each task sequentially
for i, task in enumerate(tasks):
    inefficacies = efficacies.get_inefficacies()
    neigh_sigmas = neigh_sigma_base + neigh_sigma_max * inefficacies

    print(f"Training on task {i+1}: {task}")
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loaders[i]):
            optimizer.zero_grad()
            output = model(data)
            neigh_sigma = neigh_sigmas[output.argmin(-1)].reshape(-1, 1)
            loss = lossManager.losses(
                output,
                neigh_sigma,
                anchors[target],
                anchor_sigma,
            )
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
