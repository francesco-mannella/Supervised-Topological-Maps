import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from topological_maps import TopologicalMap, som_loss

train = True

# train parameters
batch_size = 10
input_size = 3
output_size = 100
epochs = 100


def som_training(model, data_loader, epochs):
    """ Train a self-organizing map
    """

    # Initialize hyperparameters
    opt_lr = 0.02
    final_lr_prop = 0.0000001
    lr_gamma = np.exp(np.log(final_lr_prop)/epochs)
    final_std_prop = 0.000001
    std_gamma = np.exp(np.log(final_std_prop)/epochs)
    std_baseline = 1
    
    optimizer = torch.optim.Adam(som.parameters(), lr=opt_lr)

    for epoch in range(epochs):
        running_loss = 0.0
        
        std = std_baseline + model.std_init*std_gamma**epoch
        lr = model.std_init*lr_gamma**epoch 
        
        for i, data in enumerate(data_loader):
            
            inputs = data

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

if train == True:

    # Build the dataset and the data loader
    dataset = np.random.rand(1000, 3)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare the model and the optimizer
    som = TopologicalMap(input_size=input_size, output_size=output_size)

    # train
    som_training(som, dataLoader,  epochs=epochs)
    
    torch.save(som, "som_colormap.pt")

som = torch.load("som_colormap.pt")

# plot the learned weights
plt.imshow(som.weights.detach()
           .numpy()
           .reshape(3,10,10)
           .transpose(1,2,0))

# a generated color
for x in range(10):
    point = torch.rand(1, 2)*10
    col = som.backward(point).detach().numpy().ravel()
    plt.scatter(*point.detach().numpy().ravel()[::-1], fc=col, ec="black", s=100)
plt.xlim([0, 9])
plt.ylim([0, 9])
plt.show()

