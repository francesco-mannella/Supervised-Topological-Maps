import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from stm.topological_maps import TopologicalMap, som_loss
import matplotlib
from matplotlib import gridspec
matplotlib.use("agg")

def som_training(model, data_loader, epochs):
    """Train a self-organizing map.
    
    Args:
        model (TopologicalMap): The SOM model to be trained.
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

    for epoch in range(epochs):
        running_loss = 0.0
        
        std = std_baseline + model.std_init*std_gamma**epoch
        lr = model.std_init*lr_gamma**epoch 
        optimizer.param_groups[0]["lr"] = lr

        for i, data in enumerate(data_loader):

            inputs, _ = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, std)
            loss = lr*som_loss(outputs)
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
    batch_size = 100
    epochs = 400

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

        K = 1000 # enter your length here
        subsample_train_indices = torch.randperm(len(dataset))[:K]
        subset = Subset(dataset, indices=subsample_train_indices)
        dataLoader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        som = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        som_training(som, dataLoader, epochs=epochs)

        # save
        torch.save(som, "som_mnist.pt")

    som = torch.load("som_mnist.pt")

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
    

    # a generated color
    for x in range(10):
        point = torch.rand(1, 2)*10
        num = som.backward(point).detach().numpy().ravel()
        sc.set_offsets(point.detach().numpy().ravel()*28)
        img.set_array(num.reshape(28, 28))

        fig.canvas.draw()
        fig.savefig(f"stm_mnist_{x:04d}.png")

    # %%


    rdata = [d for d, l in subset]
    rdata = [rdata[38], rdata[231]]

    for i in rdata:

        _ = som(i.reshape(1, -1), .8)
        m = np.stack(som.get_representation("grid"))
        
        fig, ax = subplots(2, 1, figsize=(4, 8))
        ax[0].imshow(m.reshape(10, 10), cmap=plt.cm.binary) 
        ax[1].imshow(i.reshape(28, 28), cmap=plt.cm.gray)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()



