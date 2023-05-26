import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from stm.topological_maps import TopologicalMap, stm_loss, RadialBasis
import matplotlib
matplotlib.use("agg")

def stm_training(model, data_loader, epochs):
    """Train a supervised topological map.
    
    Args:
        model (TopologicalMap): STM model to be trained.
        data_loader (object): The data loader object used to feed data to the model.
        epochs (int): The number of epochs to train the model for.
    """

    # Initialize hyperparameters
    opt_lr = 0.5
    final_lr_prop = 1e-4
    lr_gamma = np.exp(np.log(final_lr_prop)/epochs)
    final_std_prop = 1e-4
    std_gamma = np.exp(np.log(final_std_prop)/epochs)
    std_baseline = 0.7
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

    # function to get the radial grid from a central point
    radial = RadialBasis(model.output_size, model.output_dims)

    for epoch in range(epochs):
        running_loss = 0.0
        
        std = std_baseline + model.std_init*std_gamma**epoch
        lr = model.std_init*lr_gamma**epoch 
        optimizer.param_groups[0]["lr"] = lr

        for i, data in enumerate(data_loader):

            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs, std)
            # loss depends also on radial grids centered on label points
            tags = torch.tensor(points[labels])
            rlabels = radial(tags, std, as_point=True)
            loss = lr*stm_loss(outputs, rlabels)
            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f'[{epoch}, {i:5d}] loss: {running_loss:.5f}')
            running_loss = 0.0

if __name__ == "__main__":

    train = True

    # train parameters
    input_size = 28*28
    output_size = 10*10
    batch_size = 10000
    epochs = 400

    points = np.array(
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
    )*10 

    if train == True:

        # Build the dataset and the data loader
        dataset = torchvision.datasets.MNIST(
            "/tmp/mnist",
            train=True,
            download=True,
            transform=T.Compose([
                T.ToTensor(), 
                T.Lambda(lambda x: torch.flatten(x)),
                T.Lambda(lambda x: (x - x.min())/(x.max() - x.min()))
                ]),
        )
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stm_training(som, dataLoader, epochs=epochs)

        # save
        torch.save(som, "stm_mnist.pt")

    som = torch.load("stm_mnist.pt")

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
    sc = ax1.scatter(-1,-1, fc="red", ec="white", s=100)
    ax1.set_xlim(0, 28*10)
    ax1.set_ylim(28*10, 0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_axis_off()

    ax2 = fig.add_subplot(spec[:4, 10:])
    ax2.set_xlim(0, 28)
    ax2.set_ylim(28, 0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axis_off()
    img = ax2.imshow(np.zeros([28, 28]), 
                    cmap=plt.cm.gray, vmin=0, vmax=1)
    
    ax3 = fig.add_subplot(spec[6:, 10:])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(10, 0)
    ax3.set_xticks([0, 9])
    ax3.set_yticks([0, 9])
    #ax3.set_axis_off()
    ax3.scatter(*points.T)
    for i, point in enumerate(points):
        ax3.text(*(point + [0.5, -0.5]), f'{i}', size=20, ha="center", va="center")
        ax3.text(*(point)+ [0.5, -0.5], f'{i}', size=20, ha="center", va="center")

    # a generated color
    for x in range(10):
        point = torch.rand(1, 2)*10
        num = som.backward(point).detach().numpy().ravel()
        sc.set_offsets(point.detach().numpy().ravel()*28)
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f"stm_mnist_{x:04d}.png")
