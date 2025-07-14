# %%

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from kmeans_pytorch import kmeans as KMeans
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from stm.topological_maps import LossEfficacyFactory, TopologicalMap


# %%


def create_data_loaders(train_data, test_data, task, batch_size, subset_size):
    select = [item in task for item in train_data.targets]
    indices = torch.arange(len(train_data.targets))[select]
    train_subset = Subset(train_data, indices)

    select = [item in task for item in test_data.targets]
    indices = torch.arange(len(test_data.targets))[select]
    test_subset = Subset(test_data, indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def initialize_models(input_dim, latent_dim, learning_rate):
    model = TopologicalMap(input_dim, latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


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


# %%


def to_numpy(x):
    return x.cpu().detach().numpy()


def get_regress_matrix(model, loaders):

    resps = []
    for loader in loaders:
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            norms = model(data)
            latent = torch.softmax(1000 / norms, 1).round()
            resps.append(
                np.hstack(
                    [
                        to_numpy(target).reshape(-1, 1),
                        to_numpy(latent).reshape(-1, model.output_size),
                    ]
                )
            )

    resps = pd.DataFrame(
        np.vstack(resps),
        columns=["target", *np.arange(model.side**2)],
    )

    regres_w = resps.groupby("target").mean().reset_index().to_numpy()
    regres_w[:, 1:] /= regres_w[:, 1:].sum(1).reshape(-1, 1)

    return regres_w


# %%


def evaluate_model(model, test_loaders, w_regress, DEVICE):
    correct = 0
    total = 0
    regress = w_regress[:, 1:]
    with torch.no_grad():
        for test_loader in test_loaders:
            for data, target in test_loader:

                data, target = data.to(DEVICE), target.to(DEVICE)
                norms = model(data)
                latent = torch.softmax(1000 / norms, 1)
                predicted = (to_numpy(latent) @ regress.T).argmax(1)
                total += target.size(0)
                correct += (predicted == to_numpy(target)).sum()

    accuracy = correct / total

    return accuracy


# %%


def get_groups(model, n_groups):

    weights = model.get_weights("torch").T
    weights = torch.nn.functional.normalize(weights, 0)
    group_labels, group_centroids = KMeans(
        weights,
        num_clusters=n_groups,
        tol=1e-30,
        device=DEVICE,
    )

    return group_labels, group_centroids


# %%
class Plotter:
    def __init__(self, model, loss_manager, side, imside, cmap):
        self.model = model
        self.loss_manager = loss_manager
        self.side = side
        self.imside = imside
        self.cmap = cmap
        plt.ion()
        plt.close("all")
        self.fig, self.ax = plt.subplots(1, 4)
        self.ax[0].set_axis_off()
        self.im = self.ax[0].imshow(np.zeros([imside, imside]), vmin=0, vmax=1)
        self.ax[1].set_axis_off()
        self.efim = self.ax[1].imshow(np.zeros([side, side]), vmin=0, vmax=1)
        self.ax[2].set_axis_off()
        self.clim = self.ax[2].imshow(np.zeros([side, side, 3]))
        self.ax[3].set_axis_off()
        self.bar = self.ax[3].imshow(
            cmap.reshape(-1, 1, 4) * (np.ones([2, 4]).reshape(1, 2, 4))
        )
        plt.pause(0.1)

    def plot_weights_and_efficacies(self, group_labels):
        w = (
            self.model.get_weights()
            .reshape(self.imside, self.imside, self.side, self.side)
            .transpose(2, 0, 3, 1)
            .reshape(self.imside * self.side, self.imside * self.side)
        )
        ef = self.loss_manager.get_efficacies().reshape(self.side, self.side)
        cl = group_labels.reshape(20, 20).detach().cpu().numpy()
        cl = self.cmap[cl]
        self.im.set_array(w)
        self.efim.set_array(ef)
        self.clim.set_array(cl)
        plt.pause(0.1)


if __name__ == "__main__":

    sys.stdout.flush()
    sys.stderr.flush()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NP_FORMAT = {"float": "{:8.4f}".format}

    np.set_printoptions(formatter=NP_FORMAT, linewidth=999)

    tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    learning_rate = 0.001
    batch_size = 64
    epochs = 100
    subset = -1
    input_dim = 784
    latent_dim = 20 * 20
    anchor_sigma = 1.4
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
            [9.00, 9.00],
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
            "#404040",
            "#000000",
        ]
    )(np.linspace(0, 1, 11))

    print("Initialize MNIST dataset")

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
    # %%
    print("Initialize Task Dataset loaders")
    train_loaders, test_loaders = [], []
    for task in tasks:
        train_loader, test_loader = create_data_loaders(
            mnist_train, mnist_test, task, batch_size, subset
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    model, optimizer = initialize_models(
        input_dim,
        latent_dim,
        learning_rate,
    )

    print("Initialize the LOSS manager")
    lossManager = LossEfficacyFactory(
        model=model,
        mode="stm",
        efficacy_radial_sigma=efficacy_radial_sigma,
        efficacy_decay=efficacy_decay,
        efficacy_saturation_factor=efficacy_saturation_factor,
    ).to(DEVICE)

    print("Initialize plotting")
    side = model.radial.side
    imside = 28
    # plotter = Plotter(model, lossManager, side, imside, cmap)

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

        w_regress = get_regress_matrix(model, train_loaders[: i + 1])
        accuracy = evaluate_model(
            model, test_loaders[: i + 1], w_regress, DEVICE
        )

        # plotter.plot_weights_and_efficacies(ordered_group_labels)

        print(f"Accuracy on task {i+1}: {100 * accuracy:.2f}%")
