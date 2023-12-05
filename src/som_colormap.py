import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import mkvideo
from torch.utils.data import Dataset, DataLoader
from stm.topological_maps import TopologicalMap, som_loss

rng = np.random.RandomState(4)

def som_training(model, data_loader, epochs):
    """Train a self-organizing map.
    
    Args:
        model (TopologicalMap): The SOM model to be trained.
        data_loader (object): The data loader object used to feed data to the model.
        epochs (int): The number of epochs to train the model for.
    """
    # Initialize hyperparameters
    opt_lr = 0.02
    final_lr_prop = 0.0000001
    lr_gamma = np.exp(np.log(final_lr_prop) / epochs)
    final_std_prop = 0.000001
    std_gamma = np.exp(np.log(final_std_prop) / epochs)
    std_baseline = 1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)
    
    lr_values = []
    loss_values = []
    activations_data = []
    weights_data = []

    for epoch in range(epochs):
        running_loss = 0.0
        
        std = std_baseline + model.std_init * std_gamma**epoch
        lr = model.std_init * lr_gamma**epoch 

        for i, data in enumerate(data_loader):
            inputs = data

            optimizer.zero_grad()

            outputs = model(inputs, std)
            loss = lr * som_loss(outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'[{epoch}, {i:5d}] loss: {running_loss:.5f}')
            running_loss = 0.0

        lr_values.append(lr)
        loss_values.append(running_loss)
        activations_data.append(np.stack(model.get_representation("grid")))
        weights_data.append(np.stack(model.weights.tolist()))
    
    return lr_values, loss_values, activations_data, weights_data

# Plotting functions

def plot_training_results(lr_values, loss_values, activations_data, weights_data, epochs):
    """Plot the training results.
    
    Args:
        lr_values (list): List of learning rate values over epochs.
        loss_values (list): List of loss values over epochs.
        activations_data (list): List of activation data snapshots over epochs.
        weights_data (list): List of weight data snapshots over epochs.
        epochs (int): The number of epochs the model was trained for.
    """
    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1, figsize=(4, 3))

    vm = mkvideo.vidManager(fig1, "som", ".")
    vmw = mkvideo.vidManager(fig2, "som_weights", ".")
    vmlr = mkvideo.vidManager(fig3, "som_lr", ".")
    
    for epoch in range(epochs):
        plot_activations(ax1, activations_data[epoch])
        plot_weights(ax2, weights_data[epoch])
        plot_lr(ax3, lr_values[:epoch+1], epochs)
        vm.save_frame()
        vmw.save_frame()
        vmlr.save_frame()
    
    vm.mk_video()
    vmw.mk_video()
    vmlr.mk_video()

def plot_activations(ax, r):
    r = r/r.max(1).reshape(-1, 1)
    ax.cla()
    x = rng.choice(range(len(r)))
    ax.imshow(r[x].reshape(10, 10).T, vmin=-0.2, vmax=1.0, cmap=plt.cm.binary, interpolation="none")
    ax.set_axis_off()

def plot_weights(ax, w):
    w = w.reshape(3, 10, 10)
    w = w.transpose(1,2,0)
    ax.cla()
    ax.imshow(w)
    ax.set_axis_off()

def plot_lr(ax, lrs, epochs):
    ax.cla()
    tlrs = np.zeros(epochs)
    tlrs[:len(lrs)] = np.array(lrs)/10
    tlrs[len(lrs):] = np.nan

    ax.plot(range(epochs), tlrs, c="black")
    ax.scatter(len(lrs)-1, tlrs[len(lrs)-1], s=40, c="black" )
    ax.set_xlim(-epochs*0.1, epochs*1.1)
    std = 1.1
    ax.set_ylim(-std * 0.1, std * 1.1)

def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


if __name__ == "__main__":

    train = True

    # train parameters
    batch_size = 10
    input_size = 3
    output_size = 100
    epochs = 100

    if train :

        # Build the dataset and the data loader
        dataset = np.random.rand(1000, 3)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stored_data = som_training(som, dataLoader,  epochs=epochs)
        
        torch.save(som, "som_colormap.pt")

    som = torch.load("som_colormap.pt")

    # plot the learned weights
    plt.imshow(som.weights.detach()
            .numpy()
            .reshape(3,10,10)
            .transpose(1,2,0))

    # plot generated colors
    for x in range(10):
        point = torch.rand(1, 2)*10
        projection = som.backward(point).detach().numpy().ravel()
        point = point.detach().numpy().ravel()
        plt.scatter(*point, fc=projection, ec="black", s=100)
    plt.xlim([0, 9])
    plt.ylim([0, 9])
    plt.show()
   
    som.curr_std = 1.0
    n = np.stack(som.norms.tolist())
    r = np.stack(som.get_representation("grid"))
    r = r/r.max(1).reshape(-1, 1)
    c = np.stack(som.weights[:, som.bmu].T.tolist())

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    
    ci = rng.choice(range(10), 3)
    for i in range(3):
        x = ci[i]
        axes[i, 0].imshow(c[x]*np.ones([2, 2, 3]))
        axes[i, 0].set_axis_off()
        hinton(n[x].reshape(10, 10), ax=axes[i, 1] )
        axes[i, 2].imshow(r[x].reshape(10, 10).T, vmin=-0.2, vmax=1.0, cmap=plt.cm.binary, interpolation="none")
        axes[i, 2].set_axis_off()
    fig.tight_layout(pad=1)
    plt.show()


    #

    lr_values, loss_values, activations_data, weights_data = stored_data
    plot_training_results(lr_values, loss_values, activations_data, weights_data, epochs)
