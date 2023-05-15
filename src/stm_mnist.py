import torch, torchvision
from torchvision import transforms as T
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from stm.opological_maps import TopologicalMap, stm_loss, RadialBasis


train = False

# training parameters
epochs = 40
input_size = 28*28
output_size = 10*10
batch_size = 10000

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

def stm_training(model, data_loader, epochs):
    """Train a self-organizing map.
    
    Args:
        model (object): The self-organizing map model to be trained.
        data_loader (object): The data loader object used to feed data to the model.
        epochs (int): The number of epochs to train the model for.
    """
    
    # Initialize hyperparameters
    opt_lr = 0.5
    final_lr_prop = 1e-4
    lr_gamma = np.exp(np.log(final_lr_prop)/epochs)
    final_std_prop = 1e-4
    std_gamma = np.exp(np.log(final_std_prop)/epochs)
    std_baseline = 0.6
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_lr)

    # function to get the radial grid from a central point
    radial = RadialBasis(model.output_size, model.output_dims)

    for epoch in range(epochs):
        running_loss = 0.0
        
        std = std_baseline + model.std_init*std_gamma**epoch
        lr = model.std_init*lr_gamma**epoch 

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
            print(f"[{epoch}, {i:5d}] loss: {running_loss:.3f}")
            running_loss = 0.0

if train == True:

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
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare the model and the optimizer
    stm = TopologicalMap(input_size=input_size, output_size=output_size)

    # train
    stm_training(stm, dataLoader, epochs=epochs)
    
    # save
    torch.save(stm, "stm_mnist.pt")

stm = torch.load("stm_mnist.pt")

# plot the learned weights
w = (
    stm.weights.detach()
    .numpy()
    .reshape(28, 28, 10, 10)
    .transpose(2, 0, 3, 1)
    .reshape(28 * 10, 28 * 10)
)

plt.ion()
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(131)
ax1.imshow(w, cmap=plt.cm.gray)
sc = ax1.scatter(-1, -1, fc="red", ec="white", s=100)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim(0, 10 * 28)
ax1.set_ylim(10 * 28, 0)

ax2 = fig.add_subplot(132, aspect="equal")
ax2.set_xlim(0, 10)
ax2.set_ylim(10, 0)

ax2.scatter(*points.T)
for i, (x, y) in enumerate(points):
    ax2.text(y - 0.2, x - 0.2, f"{i}", ha="center", size=12)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(0, 10)
ax2.set_ylim(10, 0)

ax3 = fig.add_subplot(133)
ax3.set_xlim(0, 28)
ax3.set_ylim(28, 0)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_axis_off()
img = ax3.imshow(np.zeros([28, 28]), cmap=plt.cm.gray, vmin=0, vmax=1)

# a generated color
for x in range(10):
    point = torch.rand(1, 2) * 10
    num = stm.backward(point).detach().numpy().ravel()
    sc.set_offsets(point.detach().numpy().ravel() * 28)
    img.set_array(num.reshape(28, 28))
    plt.pause(2)
