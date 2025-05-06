# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import Subset
from torchvision import datasets, transforms

from stm.topological_maps import LossFactory, TopologicalMap


class Encoder(nn.Module):
    def __init__(self, latent_dimension):

        super(Encoder, self).__init__()

        self.l1 = nn.conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.l2 = nn.conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.l3 = nn.linear(
            in_features=32 * 7 * 7,
            out_features=latent_dimension,
        )

    def forward(self, input_tensor):
        x = self.l1(input_tensor)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.l3(x)
        x = F.relu(x)
        return x


class MoEncoders(nn.Module):

    def __init__(self, latent_dimension, num_encoders):

        super(MoEncoders, self).__init__()
        self.encoders = [
            Encoder(latent_dimension) for i in range(num_encoders)
        ]

    def forward(self, input_tensor, gating):

        gating = F.softmax(gating)
        x_stack = torch.stack([enc(input_tensor) for enc in self.encoders])
        x = x_stack * gating.reshape(-1, 1, 1)
        x = x.mean(dim=0)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dimension):

        super(Decoder, self).__init__()
        self.l1 = nn.Linear(
            in_features=latent_dimension,
            out_features=32 * 7 * 7,
        )
        self.l2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.l3 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, latent_variable):
        z = self.l1(latent_variable)
        z = F.relu(z)
        z = z.view(z.size(0), 32, 7, 7)
        z = self.l2(z)
        z = F.relu(z)
        z = torch.sigmoid(self.decoder_deconv2(z))
        return z


class MoDecoders(nn.Module):

    def __init__(self, latent_dimension, num_decoders):

        super(MoDecoders, self).__init__()
        self.decoders = [
            Decoder(latent_dimension) for i in range(num_decoders)
        ]

    def forward(self, latent_variable, gating):

        gating = F.softmax(gating)
        x_stack = torch.stack([dec(latent_variable) for dec in self.decoders])
        x = x_stack * gating.reshape(-1, 1, 1)
        x = x.mean(dim=0)

        return x


class TopologicalAE(nn.Module):
    """
    Topological Autoencoder model for unsupervised representation learning.

    This class implements an autoencoder architecture where the encoder maps
    input data to a latent space, which is topologically constrained through a
    custom mapping, and the decoder reconstructs the input data from the latent
    space.
    """

    def __init__(self, latent_dimension, som_dimension):
        """
        Initialize the Topological Autoencoder with specified latent dimension.

        Args:
            latent_dimension (int): The dimensionality of the latent space.
            som_dimension (int): The dimensionality of the topological space.
        """
        super(TopologicalAE, self).__init__()

        self.encoder = MoEncoders(latent_dimension, 10)
        self.decoder = MoDecoders(latent_dimension, 10)

        self.topological_map = TopologicalMap(
            input_size=latent_dimension,
            output_size=som_dimension,
        )

    def forward(self, input_tensor, ):
        """
        Executes the forward pass for the autoencoder.

        Args:
            - input_tensor (torch.Tensor): The input data.

        Returns:
            - (torch.Tensor, torch.Tensor): The reconstructed output and the
              normalized code.
        """
        latent_variable = self.encode(input_tensor)
        topological_output = self.topological_map(latent_variable)
        reconstructed_output = self.decode(latent_variable)
        return reconstructed_output, topological_output


# define the loss function
class Loss:
    def __init__(self, model, mode):
        self.som_loss = LossFactory(model, mode)

    def __call__(
        self,
        x,
        reconstructed,
        topological_output,
        neighborhood_std,
        beta,
    ):
        MSE = nn.MSELoss(reduction="sum")
        reconstruction_loss = MSE(
            reconstructed, x.reshape(reconstructed.shape)
        )
        map_loss = self.som_loss.losses(topological_output, neighborhood_std)

        return reconstruction_loss + beta * map_loss


# %%

if __name__ == "__main__":

    # Define the training loop
    def training_epoch(model, train_loader, optimizer, epoch):
        model.train()
        loss_function = Loss(model.topological_map, "som")
        train_loss = 0
        neighborhood_std = 10 * (0.99**epoch)
        lr = 1 * (0.99**epoch)
        beta = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, topological_output = model(data)
            losses = loss_function(
                data,
                reconstructed,
                topological_output,
                neighborhood_std,
                beta,
            )
            losses *= lr
            loss = losses.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item() / len(data),
                    )
                )
        print(
            "====> Epoch: {} Average loss: {:.4f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset

    mnist_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms.ToTensor()
    )
    mnist_data = Subset(mnist_data, range(10))
    train_loader = torch.utils.data.DataLoader(
        mnist_data, batch_size=2, shuffle=True
    )

    # Initialize AE model
    latent_dim = 10 * 10
    model = TopologicalAE(latent_dim).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the VAE
    num_epochs = 1000
    for epoch in range(num_epochs):
        training_epoch(model, train_loader, optimizer, epoch)

    # %%

    data = np.random.rand(1000, 2)
    kmeans = KMeans(
        init="random", n_clusters=10, n_init=10, max_iter=300, random_state=42
    )
    kmeans.fit(data)
    points = torch.tensor(10 * kmeans.cluster_centers_).to(device)

    # %%

    plt.ion()

    for i, point in enumerate(points):
        z = model.topological_map.backward(points)
        generated = model.decode(z.unsqueeze(0))
        plt.subplot(1, 10, i + 1)
        plt.imshow(generated[0][0].cpu().detach().numpy())
