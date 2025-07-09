import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import tqdm
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torch_kmeans import KMeans
from torchvision import datasets, transforms

from stm.topological_maps import LossEfficacyFactory, TopologicalMap
import sys


def create_data_loaders(
    mnist_train, mnist_test, task, batch_size, subset_size
):
    train_dataset = [item for item in mnist_train if item[1] in task]
    test_dataset = [item for item in mnist_test if item[1] in task]
    subset_indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def initialize_models(input_dim, latent_dim, learning_rate, regress_lr):
    model = TopologicalMap(input_dim, latent_dim).to(DEVICE)
    regress_model = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 10),
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=1),
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    regress_optimizer = optim.Adam(regress_model.parameters(), lr=regress_lr)
    regress_loss = torch.nn.MSELoss()
    return model, regress_model, optimizer, regress_optimizer, regress_loss


def train_stm(
    model,
    optimizer,
    train_loader,
    loss_manager,
    anchors,
    epochs,
    neigh_sigma_base,
    neigh_sigma_max,
    lr_base,
    lr_max,
    anchor_sigma,
    DEVICE,
):
    for epoch in tqdm.tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)

            loss = loss_manager.loss(
                output,
                neighborhood_baseline=neigh_sigma_base,
                neighborhood_max=neigh_sigma_max,
                modulation_baseline=lr_base,
                modulation_max=lr_max,
                anchors=anchors[target],
                neighborhood_std_anchors=anchor_sigma,
            )

            loss.backward()
            optimizer.step()


def train_regressor(
    model,
    regress_model,
    regress_optimizer,
    regress_loss,
    train_loader,
    ohv_target,
    epochs,
    DEVICE,
):
    for epoch in tqdm.tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            regress_optimizer.zero_grad()
            latent = model(data)
            output = regress_model(latent)
            loss = regress_loss(output, ohv_target[target])
            loss.backward()
            regress_optimizer.step()


def evaluate_model(model, regress_model, test_loader, DEVICE):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            latent = model(data)
            output = regress_model(latent)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


class Plotter:
    def __init__(self, model, loss_manager, side, imside, cmap):
        self.model = model
        self.loss_manager = loss_manager
        self.side = side
        self.imside = imside
        self.cmap = cmap
        plt.ion()
        plt.close("all")
        self.fig, self.ax = plt.subplots(1, 3)
        self.ax[0].set_axis_off()
        self.im = self.ax[0].imshow(np.zeros([imside, imside]), vmin=0, vmax=1)
        self.ax[1].set_axis_off()
        self.efim = self.ax[1].imshow(np.zeros([side, side]), vmin=0, vmax=1)
        self.ax[2].set_axis_off()
        self.clim = self.ax[2].imshow(np.zeros([side, side, 3]))
        plt.pause(0.1)

    def plot_weights_and_efficacies(self, group_labels):
        w = (
            self.model.weights.detach()
            .cpu()
            .numpy()
            .reshape(self.imside, self.imside, self.side, self.side)
            .transpose(2, 0, 3, 1)
            .reshape(self.imside * self.side, self.imside * self.side)
        )
        ef = (
            self.loss_manager._efficacies.cpu()
            .detach()
            .numpy()
            .reshape(self.side, self.side)
        )
        cl = group_labels.reshape(20, 20).detach().cpu().numpy()
        cl = self.cmap[cl]
        self.im.set_array(w)
        self.efim.set_array(ef)
        self.clim.set_array(cl)
        plt.pause(0.1)


def cluster(data, k, max_iters=100):
    kmeans = KMeans(
        n_clusters=k,
        max_iter=max_iters,
        init="k-means++",
        n_init=10,
        random_state=0,
    )
    data = data.unsqueeze(0)
    kmeans.fit(data)
    labels = kmeans.predict(data).squeeze()
    centroids = torch.stack(
        [data[0, labels == x].mean(0) for x in labels.unique()]
    )
    return centroids, labels


if __name__ == "__main__":

    sys.stdout.flush()
    sys.stderr.flush()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NP_FORMAT = {"float": "{:8.4f}".format}

    np.set_printoptions(formatter=NP_FORMAT, linewidth=999)

    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    learning_rate = 0.001
    regress_learning_rate = 0.001
    batch_size = 64
    epochs = 100
    subset = -1
    input_dim = 784
    latent_dim = 20 * 20
    anchor_sigma = 1.2
    neigh_sigma_max = 40
    neigh_sigma_base = 0.7
    lr_max = 2
    lr_base = 0.001
    efficacy_radial_sigma = 10
    efficacy_decay = 0.005
    efficacy_saturation_factor = 2.5

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
    ).to(DEVICE) * np.sqrt(latent_dim)

    cmap = ListedColormap(
        [
            "#FF5733",
            "#33FF57",
            "#5733FF",
            "#FFFF33",
            "#33FFFF",
            "#FF33FF",
            "#FF8000",
            "#8000FF",
            "#00FF80",
            "#808080",
        ]
    )(np.linspace(0, 1, 10))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ]
    )
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loaders = []
    test_loaders = []
    for task in tasks:
        train_loader, test_loader = create_data_loaders(
            mnist_train, mnist_test, task, batch_size, subset
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    model, regress_model, optimizer, regress_optimizer, regress_loss = (
        initialize_models(
            input_dim, latent_dim, learning_rate, regress_learning_rate
        )
    )

    lossManager = LossEfficacyFactory(
        model=model,
        mode="stm",
        efficacy_radial_sigma=efficacy_radial_sigma,
        efficacy_decay=efficacy_decay,
        efficacy_saturation_factor=efficacy_saturation_factor,
    ).to(DEVICE)

    side = model.radial.side
    imside = 28
    plotter = Plotter(model, lossManager, side, imside, cmap)

    for i, task in enumerate(tasks):
        print(f"Training on task {i+1}: {task}")
        train_stm(
            model,
            optimizer,
            train_loaders[i],
            lossManager,
            anchors,
            epochs,
            neigh_sigma_base,
            neigh_sigma_max,
            lr_base,
            lr_max,
            anchor_sigma,
            DEVICE,
        )

        print(f"Training regressor on task {i+1}: {task}")
        ohv_target = torch.eye(10).to(DEVICE)
        train_regressor(
            model,
            regress_model,
            regress_optimizer,
            regress_loss,
            train_loaders[i],
            ohv_target,
            epochs,
            DEVICE,
        )

        group_centroids, group_labels = cluster(
            model.weights.T.detach(), i * 2 + 1 if i > 0 else 2
        )
        accuracy = evaluate_model(
            model, regress_model, test_loaders[i], DEVICE
        )

        plotter.plot_weights_and_efficacies(group_labels)

        print(f"Accuracy on task {i+1}: {100 * accuracy:.2f}%")
