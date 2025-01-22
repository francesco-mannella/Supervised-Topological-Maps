import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import mkvideo
from torch.utils.data import Dataset, DataLoader
from stm.topological_maps import TopologicalMap, Updater

rng = np.random.RandomState(4)


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
    loss_modulation_scale = 1
    neighborhood_std_final = 0.000001
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final) / epochs)
    neighborhood_std_baseline = 1
    neighborhood_std_scale = model.side

    # Initialize som updater 
    updater = Updater(
            model, 
            learning_rate=optimizer_learning_rate,
            mode="som")


    # Initialize lists to store output values
    modulation_rate_values = []
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
        modulation_rate = loss_modulation_scale * loss_modulation_gamma**epoch

        # Iterate over data batches
        for i, data in enumerate(data_loader):
            inputs = data

            # Forward pass through the model
            outputs = model(inputs)

            # update
            _, loss = updater(outputs, neighborhood_std, modulation_rate )

            running_loss += loss.item()

            # Print loss
            print(f"[{epoch}, {i:5d}] loss: {running_loss:.5f}")

        # Append values to corresponding lists
        modulation_rate_values.append(modulation_rate)
        loss_values.append(running_loss)
        activations_data.append(np.stack(model.get_representation(outputs, "grid")))
        weights_data.append(np.stack(model.weights.tolist()))

    # Return output values
    return modulation_rate_values, loss_values, activations_data, weights_data


def plot_training_results(
    lr_values, loss_values, activations_data, weights_data, epochs
):
    """Plot the training results.

    Args:
        lr_values (list): List of learning rate values over epochs.
        loss_values (list): List of loss values over epochs.
        activations_data (list): List of activation data snapshots over epochs.
        weights_data (list): List of weight data snapshots over epochs.
        epochs (int): The number of epochs the model was trained for.
    """
    # Create subplots for the activation, weight, and learning rate plots
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1, figsize=(4, 3))

    # Create video managers for each plot
    vm = mkvideo.vidManager(fig1, "som", ".")
    vmw = mkvideo.vidManager(fig2, "som_weights", ".")
    vmlr = mkvideo.vidManager(fig3, "som_lr", ".")

    # Iterate over each epoch
    for epoch in range(epochs):
        # Plot the activations
        plot_activations(ax1, activations_data[epoch])
        # Plot the weights
        plot_weights(ax2, weights_data[epoch])
        # Plot the learning rate
        plot_lr(ax3, lr_values[: epoch + 1], epochs)
        # Save each frame of the plots
        vm.save_frame()
        vmw.save_frame()
        vmlr.save_frame()

    # Create videos from the saved frames
    vm.mk_video()
    vmw.mk_video()
    vmlr.mk_video()


def plot_activations(ax, r):
    """Plot the activations.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        r (numpy.ndarray): The activation values.
    """
    # Normalize the activation values
    r = r / r.max(1).reshape(-1, 1)
    # Clear the plot
    ax.cla()
    # Choose a random index
    x = rng.choice(range(len(r)))
    # Plot the activations as an image
    ax.imshow(
        r[x].reshape(10, 10).T,
        vmin=-0.2,
        vmax=1.0,
        cmap=plt.cm.binary,
        interpolation="none",
    )
    # Turn off the axes
    ax.set_axis_off()


def plot_weights(ax, w):
    """Plot the weights.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        w (numpy.ndarray): The weight values.
    """
    # Reshape the weight values
    w = w.reshape(3, 10, 10)
    # Transpose the weight values
    w = w.transpose(1, 2, 0)
    # Clear the plot
    ax.cla()
    # Plot the weights
    ax.imshow(w)
    # Turn off the axes
    ax.set_axis_off()


def plot_lr(ax, lrs, epochs):
    """Plot the learning rate.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        lrs (list): The learning rate values.
        epochs (int): The number of epochs.
    """
    # Clear the plot
    ax.cla()
    # Initialize an array for the learning rate values
    tlrs = np.zeros(epochs)
    # Assign the learning rate values to the array
    tlrs[: len(lrs)] = np.array(lrs) / 10
    tlrs[len(lrs) :] = np.nan
    # Plot the learning rate values
    ax.plot(range(epochs), tlrs, c="black")
    ax.scatter(len(lrs) - 1, tlrs[len(lrs) - 1], s=40, c="black")
    # Set the x-axis limits
    ax.set_xlim(-epochs * 0.1, epochs * 1.1)
    # Set the y-axis limits
    neighborhood_std = 1.1
    ax.set_ylim(-neighborhood_std * 0.1, neighborhood_std * 1.1)


def hinton(matrix, max_weight=None, ax=None):
    """Draw a Hinton diagram for visualizing a weight matrix.

    Args:
        matrix (numpy.ndarray): The weight matrix to visualize.
        max_weight (float, optional): The maximum weight value. If not provided, it will be calculated as 2 raised to the ceiling of the logarithm base 2 of the maximum absolute weight value in the matrix.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If not provided, the current axes will be used.
    """
    # Set the axes if not provided
    ax = ax if ax is not None else plt.gca()

    # Calculate the maximum weight value if not provided
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    # Set the facecolor of the axes
    ax.patch.set_facecolor("gray")

    # Set the aspect ratio of the axes to be equal
    ax.set_aspect("equal", "box")

    # Set the major locators for x-axis and y-axis to Null (no tick labels)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Iterate through the matrix and plot rectangles for each weight value
    for (x, y), w in np.ndenumerate(matrix):
        # Set the color based on the sign of the weight
        color = "white" if w > 0 else "black"
        # Calculate the size of the rectangle based on the weight value and the maximum weight value
        size = np.sqrt(abs(w) / max_weight)
        # Create a rectangle patch for each weight value
        rect = plt.Rectangle(
            [x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color
        )
        # Add the rectangle patch to the axes
        ax.add_patch(rect)

    # Autoscale the axes
    ax.autoscale_view()
    # Invert the y-axis to match the standard Hinton diagram orientation
    ax.invert_yaxis()


def plot_weights_and_colors(som):
    """
    Plots the learned weights of a self-organizing map (SOM) as an image and generates input patterns randomly
    to demonstrate how they can be projected onto the SOM.

    Parameters:
        som (torch.nn.Module): The self-organizing map model.
    """
    fig, ax = plt.subplots()

    # Reshape and transpose the weights tensor
    weights = som.weights.detach().numpy().reshape(3, 10, 10).transpose(1, 2, 0)

    def get_projection(point):
        """
        Projects a given point onto the SOM.

        Parameters:
            point (torch.Tensor): The input point to be projected.

        Returns:
            projection (numpy.ndarray): The projected output of the SOM.
        """
        return som.backward(point).detach().numpy().ravel()

    # Plot the learned weights
    ax.imshow(weights)

    # Plot generated colors
    for _ in range(10):
        point = torch.rand(1, 2) * 10
        projection = get_projection(point)
        point = point.detach().numpy().ravel()
        ax.scatter(*point[::-1], fc=projection, ec="black", s=100)

    ax.set_xlim([0, 9])
    ax.set_ylim([0, 9])

    plt.show()


if __name__ == "__main__":

    train = True

    # train parameters
    batch_size = 10
    input_size = 3
    output_size = 100
    epochs = 100

    if train:

        # Build the dataset and the data loader
        dataset = np.random.rand(1000, 3)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stored_data = som_training(som, dataLoader, epochs=epochs)

        torch.save(som.state_dict(), "som_colormap.pt")

    else:

        som.load_state_dict(torch.load("som_colormap.pt"))



    # Plot generated pattern onto the som color manifold
    plot_weights_and_colors(som)

    # Plot training results
    lr_values, loss_values, activations_data, weights_data = stored_data
    plot_training_results(lr_values, loss_values, activations_data, weights_data, epochs)
