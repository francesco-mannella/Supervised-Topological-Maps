import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from stm.topological_maps import TopologicalMap, Updater
from matplotlib import gridspec

# matplotlib.use("agg")


def som_training(model, data_loader, epochs):
    """Train a self-organizing map.

    Args:
        model (TopologicalMap): The SOM model to be trained.
        data_loader (object): The data loader object used to feed data to the model.
        epochs (int): The number of epochs to train the model for.

    Returns:
        loss_modulation_values (list): List of learning rate values for each epoch.
        loss_values (list): List of loss values for each epoch.
        activations_data (list): List of activation data for each epoch.
        weights_data (list): List of weight data for each epoch.
    """

    # Initialize hyperparameters
    optimizer_learning_rate = 0.02
    loss_modulation_final = 1e-4
    loss_modulation_gamma = np.exp(np.log(loss_modulation_final) / epochs)
    loss_modulation_scale = 300
    neighborhood_std_final = 1e-4
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final) / epochs)
    neighborhood_std_baseline = 0.5 * np.sqrt(2)
    neighborhood_std_scale = model.side

    # Initialize som updater
    updater = Updater(model, learning_rate=optimizer_learning_rate, mode="som")

    # Initialize lists to store output values
    loss_modulation_values = []
    loss_values = []
    activations_data = []
    weights_data = []

    # Iterate over epochs
    for epoch in range(epochs):
        running_loss = 0.0

        # Calculate standard deviation for current epoch
        neighborhood_std = (
            neighborhood_std_baseline
            + neighborhood_std_scale * neighborhood_std_gamma**epoch
        )

        # Calculate learning rate for current epoch
        loss_modulation = loss_modulation_scale * loss_modulation_gamma**epoch

        # Iterate over data batches
        for i, data in enumerate(data_loader):
            inputs, _ = data

            # Forward pass through the model
            outputs = model(inputs)

            # update
            _, loss = updater(outputs, neighborhood_std, loss_modulation)

            running_loss += loss.item()
        running_loss /= i

        # Print loss
        print(f"[epoch: {epoch}] loss: {running_loss:.5f}")

        # Append values to corresponding lists
        loss_modulation_values.append(loss_modulation)
        loss_values.append(running_loss)
        activations_data.append(np.stack(model.get_representation(outputs, "grid")))
        weights_data.append(np.stack(model.weights.tolist()))

    # Return output values
    return loss_modulation_values, loss_values, activations_data, weights_data


if __name__ == "__main__":
    train = True

    # train parameters
    input_size = 28 * 28
    output_size = 10 * 10
    batch_size = 100
    epochs = 100

    if train is True:
        # Build the dataset and the data loader
        dataset = torchvision.datasets.MNIST(
            "/tmp/mnist",
            train=True,
            download=True,
            transform=T.Compose(
                [
                    T.ToTensor(),
                    T.Lambda(lambda x: torch.flatten(x)),
                    T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
                ]
            ),
        )

        K = 1000  # enter your length here
        subsample_train_indices = torch.randperm(len(dataset))[:K]
        subset = Subset(dataset, indices=subsample_train_indices)
        dataLoader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stored_data = som_training(som, dataLoader, epochs=epochs)

        # save

        torch.save(som.state_dict(), "som_mnist.pt")

    else:
        som.load_state_dict(torch.load("som_mnist.pt", weights_only=True))

    # plot the learned weights
    w = (
        som.weights.detach()
        .numpy()
        .reshape(28, 28, 10, 10)
        .transpose(3, 0, 2, 1)
        .reshape(28 * 10, 28 * 10)
    )

    # %%

    fig = plt.figure(figsize=(11, 7))
    spec = gridspec.GridSpec(ncols=14, nrows=10, figure=fig)
    ax1 = fig.add_subplot(spec[:10, :10])
    ax1.imshow(w, cmap=plt.cm.gray)
    sc = ax1.scatter(-1, -1, fc="red", ec="white", s=100)
    ax1.set_xlim(0, 28 * 10)
    ax1.set_ylim(28 * 10, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()

    ax2 = fig.add_subplot(spec[:4, 10:])
    ax2.set_xlim(0, 28)
    ax2.set_ylim(28, 0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()
    img = ax2.imshow(np.zeros([28, 28]), cmap=plt.cm.gray, vmin=0, vmax=1)

    # a generated color
    for x in range(10):
        point = torch.rand(1, 2) * 10 + 0.5
        num = som.backward(point).detach().numpy().ravel()
        sc.set_offsets(point.detach().numpy().ravel() * 28)
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f"som_mnist_{x:04d}.png")

    plt.show()
