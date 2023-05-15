import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from stm.topological_maps import TopologicalMap, RadialBasis, stm_loss

train = True

# training parameters
batch_size = 50
input_size = 3
output_size = 100
epochs = 100

def stm_training(model, data_loader, epochs):
    """Train a supervised topological map.
    
    Args:
        model (object): The model to be trained.
        data_loader (object): The data loader to be used.
        epochs (int): The number of epochs to train for.
    """
    
    # Initialize hyperparameters
    opt_lr = 0.2
    final_lr_prop = 1e-8
    lr_gamma = np.exp(np.log(final_lr_prop)/epochs)
    final_std_prop = 1e-8
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
            rlabels = radial(labels.reshape(-1), 0.3*model.std_init )
            loss = lr*stm_loss(outputs, rlabels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(f"[{epoch}, {i:5d}] loss: {running_loss:.6f}")
            running_loss = 0.0

class ColorDataset(Dataset):
    """
        This dataset contains labeled colors that have been grouped into the six basic colors.
    """

    def __init__(self, size=1000):
        """
        Args:
            size (int): The maximum size of the stack. Defaults to 1000.
        """
        self.colors = np.array(
            [
                [0, 1, 0], # green
                [0, 0, 1], # blue 
                [1, 0, 0], # red
                [0, 1, 1], # cyan
                [1, 1, 0], # yellow
                [1, 0, 1], # violet
            ]
        )
        self.grid = np.array([
            [5, 1],
            [1, 8],
            [8, 8],
            [8, 3],
            [2, 3],
            [5, 7],
        ])
        
        points = np.zeros([size, 3])
        for i in range(size):
            k = i % 6
            points[i, :] = np.maximum(0, np.minimum(1, self.colors[k] + 0.2*np.random.randn(3)))

        igrid = np.array([x * 10 + y for x, y in self.grid])
        labels = np.expand_dims(points, 1) - np.expand_dims(self.colors, 0)
        labels = np.linalg.norm(labels, axis=-1)
        labels = np.argmin(labels, axis=-1)
        labels = igrid[labels]

        self.data = np.hstack([points, labels.reshape(-1, 1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            This method retrieves an item from the list of items.

            Args:
                idx (int): The index of the item to retrieve.
        """
        sample = (
            torch.from_numpy(self.data[idx, :3]),
            torch.from_numpy(self.data[idx, 3:]),
        )
        return sample

if train == True:

    # Build the dataset and the data loader
    dataset = ColorDataset(1000)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare the model and the optimizer
    stm = TopologicalMap(input_size=input_size, output_size=output_size)

    # train
    stm_training(stm, dataLoader, epochs=epochs)
    
    torch.save(stm, "stm_colormap.pt")

stm = torch.load("stm_colormap.pt")

# %%

weights = stm.weights.detach()
 

# plot the learned weights
plt.imshow(stm.weights.detach()
           .numpy()
           .reshape(3,10,10)
           .transpose(1,2,0))

# plot the label targets
plt.scatter(*dataset.grid.T[::-1], color=dataset.colors, ec="black", lw=3, s=200) 

# a generated color
for x in range(10):
    point = torch.rand(1, 2)*10
    projection = stm.backward(point).detach().numpy().ravel()
    point = point.detach().numpy().ravel()
    plt.scatter(*point, fc=projection, ec="black", s=100)
plt.xlim([0, 9])
plt.ylim([0, 9])
plt.show()

