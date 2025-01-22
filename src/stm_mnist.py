import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader, Subset
from stm.topological_maps import TopologicalMap, RadialBasis, som_stm_loss
import matplotlib

# matplotlib.use("agg")


def stm_training(model, data_loader, epochs, lr=2.0, final=1e-2, normalized_kernel=True):
    """Train a supervised topological map.

    Args:
        model (TopologicalMap): Instance of the TopologicalMap class to be trained.
        data_loader (torch.utils.DataLoader): Data loader containing training data.
        epochs (int): Number of epochs to train the model for.

    Returns:
        lr_values (list): Learning rates used in each epoch.
        loss_values (list): Loss values for each epoch.
        activations_data (list): Activation data obtained during training.
        weights_data (list): Model weights at each epoch.
    """

    # Initialize hyperparameters
    optimizer_learning_rate = lr
    loss_modulation_final = final
    loss_modulation_gamma = np.exp(np.log(loss_modulation_final) / epochs)
    neighborhood_std_final = final
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final) / epochs)
    neighborhood_std_baseline = 0.7

    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_learning_rate
    )

    # Initialize lists to store output values
    lr_values = []
    loss_values = []
    activations_data = []
    weights_data = []

    # function to get the radial grid from a central point
    # radial = RadialBasis(model.output_size, model.output_dims)

    for epoch in range(epochs):
        running_loss = 0.0

        # Calculate standard deviation for current epoch
        neighborhood_std = (
            neighborhood_std_baseline
            + model.std_init * neighborhood_std_gamma**epoch
        )

        # Calculate learning rate for current epoch
        lr = loss_modulation_gamma**epoch

        # Iterate over data batches
        for i, data in enumerate(data_loader):

            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass through the model
            outputs = model(inputs)
            # loss depends also on radial grids centered on label points
            anchors = torch.tensor(points[labels]).detach()
            # anchors = torch.tensor(points[labels])

            # calculate loss
            # rlabels = radial(anchors, neighborhood_std, as_point=True)
            # stmloss = stm_loss(outputs, rlabels)
            stmloss = som_stm_loss(som, outputs, neighborhood_std, anchors=anchors, neighborhood_std_anchors=1, normalized_kernel=normalized_kernel)
            loss = lr * stmloss

            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += stmloss.item()
            print(f'[{epoch}, {i:5d}] loss: {running_loss:.6f}')

        # Append values to corresponding lists
        lr_values.append(lr)
        loss_values.append(running_loss)
        # activations_data.append(np.stack(model.get_representation("grid")))
        # weights_data.append(np.stack(model.weights.tolist()))

    # Return output values
    return lr_values, loss_values, activations_data, weights_data


if __name__ == '__main__':

    train = True

    # train parameters
    input_size = 28 * 28
    output_size = 10 * 10
    batch_size = 10000
    epochs = 100
    final = 1e-3

    points = (
        np.array(
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

        partial_dataset = False 
        if partial_dataset:
            K = 1000
            subsample_train_indices = torch.randperm(len(dataset))[:K]
            subset = Subset(dataset, indices=subsample_train_indices)
            dataLoader = DataLoader(
                subset, batch_size=batch_size, shuffle=True
            )
        else:
            dataLoader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        lr_values, loss_values, _, _ = stm_training(
            som, dataLoader, epochs=epochs, lr=0.09, final=final, normalized_kernel=True,
        )

        # save
        torch.save(som, 'stm_mnist.pt')

        # plot loss
        fig, ax = plt.subplots()
        ax.plot(loss_values)
        fig.savefig(f'loss.png')

    som = torch.load('stm_mnist.pt')

    # plot the learned weights
    w = (
        som.weights.detach()
        .numpy()
        .reshape(28, 28, 10, 10)
        .transpose(2, 0, 3, 1)
        .reshape(28 * 10, 28 * 10)
    )

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

    ax3 = fig.add_subplot(spec[6:, 10:])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(10, 0)
    ax3.set_xticks([0, 9])
    ax3.set_yticks([0, 9])
    ax3.set_axis_off()

    ax3.scatter(points.T[1], points.T[0])
    y = np.flip(points, axis=1)
    for i, point in enumerate(y):
        ax3.text(
            *(point) + [0.5, -0.5], f'{i}', size=20, ha='center', va='center'
        )

    # a generated color
    for x in range(10):
        point = torch.rand(1, 2) * 10
        num = som.backward(point).detach().numpy().ravel()

        sc.set_offsets([point[0][1] * 28, point[0][0] * 28])
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f'stm_mnist_{x:04d}.png')
