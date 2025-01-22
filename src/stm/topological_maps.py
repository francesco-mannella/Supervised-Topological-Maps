import torch
import math
import numpy as np


class DimensionalityError(Exception):
    def __str__(self):
        return 'Dimensionality of the output must be 1D or 2D.'


class RadialBasis:
    """
    This code creates a radial grid in either 1D or 2D based on a given
    centroid. It can be used to generate a grid of points around a central
    point.
    """

    def __init__(self, size, dims):
        """
        This function creates a radial grid with a given size and dimensionality.

        Args:
            size (int): The size of the radial grid.
            dims (int): The dimensionality of the grid (1 for 1D, 2 for 2D).

        """
        self.dims = dims
        self.size = size

        if self.dims == 1:
            self.grid = torch.arange(self.size)
            self.side = self.size
        elif self.dims == 2:
            self.side = int(math.sqrt(self.size))
            if self.side**2 != self.size:
                raise 'Dimensions must be equal'
            t = torch.arange(self.side)
            meshgrids = torch.meshgrid(t, t, indexing='ij')
            self.grid = torch.stack([x.reshape(-1) for x in meshgrids]).T
        else:
            raise DimensionalityError()
        self.grid = self.grid.unsqueeze(dim=0).float()

    def __call__(self, index, std, as_point=False):
        """
        Args:
            index (int): indicates the point at the center of the Gaussian on the flattened grid, arranged in rows.
            std (float): The standard deviation of the function.
            as_point (bool, optional): Whether to treat index as a point or not. Defaults to False.

        Returns:
            The result of the function call.
        """

        if self.dims == 1:
            x = index.unsqueeze(dim=-1)
            dists = self.grid - x
        elif self.dims == 2:
            if as_point:
                x = index.unsqueeze(dim=1)
            else:
                row = index // self.side
                col = index % self.side
                x = torch.stack([row, col]).T
                x = x.unsqueeze(dim=1)
            # print(self.grid)
            dists = torch.norm(self.grid - x, dim=-1)
        """
        elif self.dims == 2:
            if as_point:
                index = index.reshape(-1, 2)
                col, row = index[:, 0], index[:, 1]
            else:
                row = index // self.side
                col = index % self.side
            x = torch.stack([row, col]).T
            x = x.unsqueeze(dim=1)
            #print(self.grid)
            dists = torch.norm(
                    self.grid - x, 
                    dim=-1)
        """
        output = torch.exp(-0.5 * (std**-2) * dists**2)
        output /= output.sum(dim=-1).unsqueeze(dim=-1)

        return output


class TopologicalMap(torch.nn.Module):
    """
    A neural network module that represents a topological map. This map models data topology
    through connected nodes
    """

    def __init__(
        self, input_size, output_size, output_dims=2, parameters=None
    ):
        """
        Initializes the TopologicalMap with specified input and output dimensions and optionally custom parameters.

        Args:
            input_size (int): Number of inputs to the network.
            output_size (int): Number of outputs from the network.
            output_dims (int, optional): Dimensionality of the output space. Defaults to 2.
            parameters (numpy.ndarray, optional): Initial weight parameters; defaults to None, which 
                                                  initializes weights with Xavier normalization.
        """

        super(TopologicalMap, self).__init__()

        if parameters is None:
            weights = torch.empty(input_size, output_size)
            torch.nn.init.xavier_normal_(weights)
            self.weights = torch.nn.Parameter(weights, requires_grad=True)
        else:
            parameters = torch.tensor(parameters).float()
            self.weights = torch.nn.Parameter(parameters, requires_grad=True)

        self.input_size = input_size
        self.output_size = output_size
        self.output_dims = output_dims
        self.radial = RadialBasis(output_size, output_dims)
        self.std_init = (
            self.output_size
            if output_dims == 1
            else int(math.sqrt(self.output_size))
        )
        self.curr_std = self.std_init
        self.bmu = None
        self.side = (
            None if output_dims == 1 else int(math.sqrt(self.output_size))
        )

    def forward(self, x):
        """
        Computes the forward pass by calculating squared Euclidean distances 
        between input data and the network's weights.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Squared Euclidean distances for each weight vector.
        """

        diffs = self.weights.unsqueeze(dim=0) - x.unsqueeze(dim=-1)

        # Compute the Euclidean norms of the differences along dimension 1
        norms = torch.norm(diffs, dim=1)

        # Square the norms to obtain squared distances
        norms2 = torch.pow(norms, 2)

        return norms2

    def find_bmu(self, x):
        """
        Identifies the index of the Best Matching Unit (BMU) for a given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Index of the BMU.
        """
        return torch.argmin(x, dim=-1).detach()


    def get_representation(self, x, rtype='point', std=None):
        """
        Generates the representation of the Best Matching Unit (BMU) based on the specified type requested.

        Args:
            x (torch.Tensor): Input tensor.
            rtype (str, optional): Type of the representation. Supported types are 'point' (default) and 'grid'.
            std (float, optional): Standard deviation for the neighborhood function. Defaults to current std.

        Returns:
            torch.Tensor or None: BMU representation based on the specified type; None if BMU is unavailable.
        """

        self.bmu = self.find_bmu(x)
        if self.bmu is not None:
            if rtype == 'point':
                if self.output_dims == 1:
                    return self.bmu.float()

                elif self.output_dims == 2:
                    row = self.bmu // self.side
                    col = self.bmu % self.side
                    return torch.stack([row, col]).T.float()

            elif rtype == 'grid':
                if std is None:
                    std = self.curr_std
                phi = self.radial(self.bmu, std)
                return phi
        else:
            return None

    def backward(self, point, std=None):
        """
        Executes the backward pass for a specified point.

        Args:
            point (int): Target point for the backward pass.
            std (float, optional): Standard deviation for the radial basis function. Defaults to current std.

        Returns:
            torch.Tensor: Result of the backward pass for the specified point.
        """

        if std is None:
            std = self.curr_std
        phi = self.radial(point, std, as_point=True)
        output = torch.matmul(phi, self.weights.T)
        return output

class Updater:
    """
    Class for updating a SOM or STM model.

    Parameters:
    model (torch model): The SOM or STM model to be updated.
    learning_rate (float): The learning rate used by the optimizer.
    mode (str): The type of update ('som' or 'stm').
    normalized_kernel (bool, optional): If the kernel is normalized. Default is True.
    """

    def __init__(self, model, learning_rate, mode='som', normalized_kernel=True):
        self.model = model
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        self.mode = mode
        self.normalized_kernel = normalized_kernel

    def loss(self, norms2, std, tags=None, std_tags=None):
        """
        Compute the SOM/STM loss.

        Parameters:
        norms2 (array-like): The squared norm of some input data.
        std (float): The standard deviation for radial calculation.
        tags (array-like, optional): Labels or tags for additional calculations. Default is None.
        std_tags (float, optional): The standard deviation for tags calculations. Default is std.

        Returns:
        float: The mean value of the computed loss.
        """
        # If tags are not provided, calculate loss without tags
        if tags is None:
            self.model.bmu = self.model.find_bmu(norms2)
            phi = self.model.radial(self.model.bmu, std)
            self.model.curr_std = std
            output = 0.5 * norms2 * phi
        # If tags are provided, incorporate them into the loss calculation
        else:
            self.model.bmu = self.model.find_bmu(norms2)
            phi = self.model.radial(self.model.bmu, std)
            if std_tags is None:
                std_tags = std
            rlabels = self.model.radial(tags, std_tags, as_point=True)
            self.model.curr_std = std
            phi_rlabels = phi * rlabels
            if self.normalized_kernel:
                phi_rlabels = phi_rlabels / phi_rlabels.amax(axis=0)
            output = 0.5 * norms2 * phi_rlabels

        return output.mean()

    def __call__(self, output, std, learning_modulation, target=None, target_std=None):
        if self.mode == 'som':
            loss = self.loss(output, std)
        elif self.mode == 'stm':
            loss = self.loss(output, std, target, target_std)
        else:
            raise ValueError("Invalid mode. Use 'som' or 'stm'.")

        loss = learning_modulation * loss
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()
