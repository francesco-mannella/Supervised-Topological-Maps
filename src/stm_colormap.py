import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from stm.topological_maps import TopologicalMap, RadialBasis, stm_loss


def stm_training(model, data_loader, epochs):
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
    optimizer_learning_rate = 0.2
    loss_modulation_final = 1e-8
    loss_modulation_gamma = np.exp(np.log(loss_modulation_final)/epochs)
    neighborhood_std_final = 1e-8
    neighborhood_std_gamma = np.exp(np.log(neighborhood_std_final)/epochs)
    neighborhood_std_baseline = 0.6

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_learning_rate)
    
    # Initialize lists to store output values
    lr_values = []
    loss_values = []
    activations_data = []
    weights_data = []

    # function to get the radial grid from a central point
    radial = RadialBasis(model.output_size, model.output_dims)

    for epoch in range(epochs):
        running_loss = 0.0

        std = neighborhood_std_baseline + model.std_init*neighborhood_std_gamma**epoch
        lr = model.std_init*loss_modulation_gamma**epoch 

        for i, data in enumerate(data_loader):

            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs, std)
            # loss depends also on radial grids centered on label points
            rlabels = radial(labels, 0.1*model.std_init, as_point=True )
            stmloss = stm_loss(outputs, rlabels)
            loss = lr*stmloss

            # backward + optimize
            loss.backward()
            optimizer.step()

            # print statistics

            running_loss += stmloss.item()
            print(f"[{epoch}, {i:5d}] loss: {running_loss:.6f}")


        # Append values to corresponding lists
        lr_values.append(lr)
        loss_values.append(running_loss)
        activations_data.append(np.stack(model.get_representation("grid")))
        weights_data.append(np.stack(model.weights.tolist()))
    
    # Return output values
    return lr_values, loss_values, activations_data, weights_data

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
        
        labels = points.reshape(-1, 1, 3) - self.colors.reshape(1, -1, 3)
        labels = np.linalg.norm(labels, axis=2)
        idcs = np.argmin(labels, axis=1)
        labels = self.grid[idcs,:]
        self.labels = labels
        self.data = points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            This method retrieves an item from the list of items.

            Args:
                idx (int): The index of the item to retrieve.
        """
        sample = (
                torch.from_numpy(self.data[idx, :]),
                torch.from_numpy(self.labels[idx, :]),
        )
        return sample

if __name__ == "__main__":

    train = True

    # training parameters
    batch_size = 50
    input_size = 3
    output_size = 100
    epochs = 100

    if train == True:

        # Build the dataset and the data loader
        dataset = ColorDataset(1000)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # prepare the model and the optimizer
        stm = TopologicalMap(input_size=input_size, output_size=output_size)

        # train
        stored_data = stm_training(stm, dataLoader, epochs=epochs)
        
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
    plt.scatter(*dataset.grid.T, color=dataset.colors, ec="black", lw=3, s=200) 

    # a generated color
    for x in range(10):
        point = torch.rand(1, 2)*10
        projection = stm.backward(point).detach().numpy().ravel()
        point = point.detach().numpy().ravel()
        plt.scatter(*point, fc=projection, ec="black", s=100)
    plt.xlim([0, 9])
    plt.ylim([0, 9])
    plt.draw()
    plt.savefig("stm_colormap.png")

