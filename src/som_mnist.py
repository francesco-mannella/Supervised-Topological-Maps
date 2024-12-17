import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from stm.topological_maps import TopologicalMap, som_stm_loss
import matplotlib
from matplotlib import gridspec

# matplotlib.use("agg")


def som_training(model, data_loader, epochs):
    """Train a self-organizing map.

    Args:
        model (TopologicalMap): The SOM model to be trained.
        data_loader (object): The data loader object used to feed data to the model.
        epochs (int): The number of epochs to train the model for.

    Returns:
        lr_values (list): List of learning rate values for each epoch.
        loss_values (list): List of loss values for each epoch.
        activations_data (list): List of activation data for each epoch.
        weights_data (list): List of weight data for each epoch.
    """

    # Initialize hyperparameters
    optimizer_learning_rate = 0.02
    loss_modulation_final = 0.0000001
    loss_modulation_gamma = np.exp(np.log(loss_modulation_final) / epochs)
    neighborhood_std_final = 0.000001
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final) / epochs)
    neighborhood_std_baseline = 1

    # Initialize optimizer for model parameters
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_learning_rate
    )

    # Initialize lists to store output values
    lr_values = []
    loss_values = []
    activations_data = []
    weights_data = []

    # Iterate over epochs
    for epoch in range(epochs):
        running_loss = 0.0

        # Calculate standard deviation for current epoch
        std = (
            neighborhood_std_baseline
            + model.std_init * neighborhood_std_gamma**epoch
        )

        # Calculate learning rate for current epoch
        lr = model.std_init * loss_modulation_gamma**epoch

        # Iterate over data batches
        for i, data in enumerate(data_loader):
            inputs, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through the model
            # outputs = model(inputs, std)
            outputs = model(inputs)

            # Calculate loss
            # sloss =  som_loss(outputs)
            sloss = som_stm_loss(som, outputs, std, tags=None)
            loss = lr * sloss

            # Backward pass and update gradients
            loss.backward()
            optimizer.step()

            running_loss += sloss.item()

            # Print loss
            print(f'[{epoch}, {i:5d}] loss: {running_loss:.5f}')

        # Append values to corresponding lists
        lr_values.append(lr)
        loss_values.append(running_loss)
        activations_data.append(np.stack(model.get_representation('grid')))
        weights_data.append(np.stack(model.weights.tolist()))

    # Return output values
    return lr_values, loss_values, activations_data, weights_data


if __name__ == '__main__':

    train = True

    # train parameters
    input_size = 28 * 28
    output_size = 10 * 10
    batch_size = 100
    epochs = 400

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

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stored_data = som_training(som, dataLoader, epochs=epochs)

        # save
        torch.save(som, 'som_mnist.pt')

    som = torch.load('som_mnist.pt')

    # plot the learned weights
    w = (
        som.weights.detach()
        .numpy()
        .reshape(28, 28, 10, 10)
        .transpose(2, 0, 3, 1)
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
        point = torch.rand(1, 2) * 10
        num = som.backward(point).detach().numpy().ravel()
        sc.set_offsets(point.detach().numpy().ravel() * 28)
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f'stm_mnist_{x:04d}.png')

    # %%

    rdata = [d for d, l in subset]
    rdata = [rdata[38], rdata[231]]

    for i in rdata:

        # _ = som(i.reshape(1, -1), .8)
        som.std = 0.8
        norms2 = som(i.reshape(1, -1))
        som.bmu = som.find_bmu(norms2)
        m = np.stack(som.get_representation('grid'))

        fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        ax[0].imshow(m.reshape(10, 10), cmap=plt.cm.binary)
        ax[1].imshow(i.reshape(28, 28), cmap=plt.cm.gray)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
