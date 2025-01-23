import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Subset
import math
import numpy as np
import torch.nn.functional as F

from topological_maps import TopologicalMap, som_loss

class TopologicalAE(nn.Module):
    def __init__(self, latent_dim):
        super(TopologicalAE, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc = TopologicalMap(256, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, 256)
        self.fc3 = nn.Linear(256, 32 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x, neighborhood_std):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        norm2_phi = self.fc(x, neighborhood_std)
        radial = self.fc.get_representation("grid")
        return norm2_phi, radial

    def reparameterize(self, radial):
        z = radial
        return z

    def decode(self, z):
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), 32, 7, 7)
        z = F.relu(self.deconv1(z))
        z = torch.sigmoid(self.deconv2(z))
        return z

    def forward(self, x, neighborhood_std):
        n2_phi, z = self.encode(x, neighborhood_std)
        z = self.reparameterize(z)
        recon_x = self.decode(z)
        return recon_x, n2_phi

        
# define the loss function
def loss_function(x, reconstructed, n2_phi):
    MSE = nn.MSELoss(reduction='sum')
    reconstruction_loss = MSE(reconstructed, x.reshape(reconstructed.shape))
    map_loss = som_loss(n2_phi) 

    return reconstruction_loss #+ map_loss

# Define the training loop
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    neighborhood_std = 10*(0.9**epoch)
    lr = 1*(0.9**epoch)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstructed, n2_phi = model(data, neighborhood_std)
        loss = loss_function(data, reconstructed, n2_phi)
        loss *= lr
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset


mnist_data = datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor())
mnist_data = Subset(mnist_data, range(10))
train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=2, shuffle=True)

# Initialize VAE model
latent_dim = 10*10
model = TopologicalAE(latent_dim).to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Train the VAE
num_epochs = 100
for epoch in range(num_epochs):
    train(model, train_loader, optimizer, epoch)

# %%


from sklearn.cluster import KMeans

data = np.random.rand(1000, 2)
kmeans = KMeans(
    init="random",
    n_clusters=10,
    n_init=10,
    max_iter=300,
    random_state=42
    )
kmeans.fit(data)
points = kmeans.cluster_centers_


# %%

import matplotlib.pyplot as plt
plt.ion()

for i, point in enumerate(points):
    norm = model.fc.radial(torch.randint(0, 10, [2]), 0.8)
    z = model.reparameterize(norm)
    z.shape
    generated = model.decode(z)
    plt.subplot(1, 10, i+1)
    plt.imshow(generated[0][0].detach().numpy())

