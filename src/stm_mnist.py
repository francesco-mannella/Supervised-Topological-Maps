import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from stm.topological_maps import TopologicalMap, Updater
import matplotlib
from matplotlib import gridspec

# matplotlib.use("agg")


def stm_training(model, data_loader, epochs):
    """Train a self-organizing map.

    Args:
        model (TopologicalMap): Instance of the TopologicalMap class to be trained.
        data_loader (torch.utils.DataLoader): Data loader containing training data.
        epochs (int): Number of epochs to train the model for.

    Returns:
        loss_modulation_values (list): Learning rates used in each epoch.
        loss_values (list): Loss values for each epoch.
        activations_data (list): Activation data obtained during training.
        weights_data (list): Model weights at each epoch.
    """
    
    # Initialize hyperparameters
    optimizer_learning_rate = 0.02
    loss_modulation_final = 1e-4
    loss_modulation_gamma = np.exp(np.log(loss_modulation_final) / epochs)
    loss_modulation_scale = 100
    neighborhood_std_final = 1e-4
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final) / epochs)
    neighborhood_std_baseline = 0.5*np.sqrt(2)
    neighborhood_std_scale = model.side
    
    points = (
        torch.tensor(
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
        )
        * 10
    )
    

    # Initialize stm updater 
    updater = Updater(
            model, 
            learning_rate=optimizer_learning_rate,
            mode="stm")

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
            inputs, labels = data

            # Forward pass through the model
            outputs = model(inputs)

            # update
            _, loss = updater(
                    outputs, 
                    neighborhood_std, 
                    loss_modulation, 
                    anchors=points[labels],
                    neighborhood_std_anchors=neighborhood_std_baseline)

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


if __name__ == '__main__':

    train = True
    # train = False 

    # train parameters
    input_size = 28 * 28
    output_size = 10 * 10
    batch_size = 100
    epochs = 500

    # prepare the model and the optimizer
    stm = TopologicalMap(input_size=input_size, output_size=output_size)


    if train == True:

        # Build the dataset and the data loader
        dataset = torchvision.datasets.MNIST(
            '/tmp/mnist',
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

        K = 1000   # enter your length here
        subsample_train_indices = torch.randperm(len(dataset))[:K]
        subset = Subset(dataset, indices=subsample_train_indices)
        dataLoader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # train
        stored_data = stm_training(stm, dataLoader, epochs=epochs)

        # save

        torch.save(stm.state_dict(), "stm_mnist.pt")

    else:

        stm.load_state_dict(torch.load("stm_mnist.pt", weights_only=True))


    # plot the learned weights
    w = (
        stm.weights.detach()
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
    sc = ax1.scatter(-1, -1, fc='red', ec='white', s=100)
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
        num = stm.backward(point, 0.5*np.sqrt(2)).detach().numpy().ravel()
        sc.set_offsets(point.detach().numpy().ravel() * 28)
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f'stm_mnist_{x:04d}.png')
    
    plt.show()

