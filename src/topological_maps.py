import torch
import math
import numpy as np


class DimensionalityError(Exception):
    def __str__(self):
        return "Dimensionality of the output must be 1D or 2D."


class RadialBasis:
    """build radial grids in 1D or 2D based on a centroid"""

    def __init__(self, size, dims):
        """
        Arguments:
            size: size of the radial grid
            dims: dimensionality of the grid.
                if 1, it is a 1D grid. if 2, it is a 2D grid.
        """

        self.dims = dims
        self.size = size

        if self.dims == 1:
            self.grid = torch.arange(self.size)
            self.side = self.size
        elif self.dims == 2:
            self.side = int(math.sqrt(self.size))
            t =  torch.arange(self.side)
            meshgrids = torch.meshgrid(t, t)
            self.grid = torch.stack([x.reshape(-1) for x in meshgrids]).T
        else:
            raise DimensionalityError()
        self.grid = self.grid.unsqueeze(dim=0).float()

    def __call__(self, index, std, as_point=False):
        if self.dims == 1:
            x = index.unsqueeze(dim=-1)
            dists = self.grid - x
        elif self.dims == 2:
            if as_point:
                col, row = index[:, 0], index[:, 1]
            else:
                row = index // self.side
                col = index % self.side
            x = torch.stack([row, col]).T
            x = x.unsqueeze(dim=1)
            dists = torch.norm(
                    self.grid - x, 
                    dim=-1)
        output = torch.exp(-0.5 * (std**-2) * dists**2)
        output /= output.sum(dim=-1).unsqueeze(dim=-1)

        return output

class TopologicalMap(torch.nn.Module):
    """A topological map"""

    def __init__(self, input_size, output_size, output_dims=2):
        """
        Arguments:
            input_size: size of the input patern
            output_size: size of the output layer
            output_dims: dimensionality of the output layer.
                if 1, it is a 1D grid. if 2, it is a 2D grid.
            init_lr: initial learning rate
        """

        super(TopologicalMap, self).__init__()

        self.weights = torch.nn.Parameter(torch.randn(input_size, output_size), 
                                          requires_grad=True)
        self.input_size = input_size
        self.output_size = output_size
        self.output_dims = output_dims
        self.radial = RadialBasis(output_size, output_dims)
        self.std_init = (self.output_size if output_dims == 1 
                         else int(math.sqrt(self.output_size)))
        self.curr_std = self.std_init

    def forward(self, x, std):
        """ Activation of the region of the output layer around the best-matching unit 
        """
        diffs = self.weights.unsqueeze(dim=0) - x.unsqueeze(dim=-1)
        norms = torch.norm(diffs, dim=1)
        norms2 = torch.pow(norms, 2)
        idx = torch.argmin(norms, dim=-1).detach()
        phi = self.radial(idx, std)
        self.curr_std = std

        return norms2*phi

    def backward(self, point, std=None):

        if std is None: std = self.curr_std
        phi = self.radial(point, std, as_point=True)
        output = torch.matmul(phi, self.weights.T).T
        return  output



def som_loss(output):
    """ Loss function to minimize for reproducing a Self-Organizing Map
    """
    return 0.5*output.mean()

def stm_loss(output, target):
    """ Loss function to minimize for reproducing a Supervised Topological Map
    """
    filtered = output * target
    return 0.5*filtered.mean()


