import math

import torch


class DimensionalityError(Exception):
    def __str__(self):
        return "Dimensionality of the output must be 1D or 2D."


class RadialBasis(torch.nn.Module):
    """
    This code creates a radial grid in either 1D or 2D based on a given
    centroid. It can be used to generate a grid of points around a central
    point.
    """

    def __init__(self, size, dims):
        """
        This function creates a radial grid with a given size and
        dimensionality.

        Args:
            size (int): The size of the radial grid.
            dims (int): The dimensionality of the grid (1 for 1D, 2 for 2D).

        """
        super(RadialBasis, self).__init__()
        self.dims = dims
        self.size = size

        if self.dims == 1:
            self.grid = torch.arange(self.size)
            self.side = self.size
        elif self.dims == 2:
            self.side = int(math.sqrt(self.size))
            if self.side**2 != self.size:
                raise "Dimensions must be equal"
            t = torch.arange(self.side)
            meshgrids = torch.meshgrid(t, t, indexing="ij")
            self.grid = torch.stack([x.reshape(-1) for x in meshgrids]).T
        else:
            raise DimensionalityError()
        self.grid = self.grid.unsqueeze(dim=0).float()

    def _apply(self, fn):
        """
        Applies a function recursively to all tensors that are sub-modules or
        direct attributes of this module.

        This method overrides the `_apply` method of `nn.Module` to ensure that
        the function `fn` is also applied to the `radial` attribute, which is
        assumed to be a `nn.Module` as well. This is necessary to propagate
        changes like moving tensors to a different device
        (e.g., CPU to GPU) or changing their data type.
        """
        super(RadialBasis, self)._apply(fn)
        self.grid = fn(self.grid)
        return self

    def forward(self, index, std, as_point=False):
        """
        Args:
            - index (int): Indicates the point at the center of the Gaussian on
              the flattened grid, which is arranged in rows.
            - std (float): The standard deviation of the function.
            - as_point (bool, optional): Whether to treat index as a point or
              not.  Defaults to False.

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
                x = torch.stack([row, col]).T.to(device=self.grid.device)
                x = x.unsqueeze(dim=1)
            dists = torch.norm(self.grid - x, dim=-1)

        output = torch.exp(-0.5 * (std**-2) * dists**2)
        output /= output.sum(dim=-1).unsqueeze(dim=-1)

        return output


class TopologicalMap(torch.nn.Module):
    """
    A neural network module that represents a topological map. This map models
    data topology through connected nodes
    """

    def __init__(
        self, input_size, output_size, output_dims=2, parameters=None
    ):
        """
        Initializes the TopologicalMap with specified input and output
        dimensions and optionally custom parameters.

        Args:
            - input_size (int): Number of inputs to the network.
            - output_size (int): Number of outputs from the network.
            - output_dims (int, optional): Dimensionality of the output space.
              Defaults to 2.
            - parameters (numpy.ndarray, optional): Initial weight parameters;
              defaults to None, which initializes weights with Xavier
              normalization.
        """

        super(TopologicalMap, self).__init__()

        if parameters is None:
            weights = torch.empty(input_size, output_size)
            torch.nn.init.xavier_normal_(weights)
            self.weights = torch.nn.Parameter(
                1e-4 * weights, requires_grad=True
            )
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

    def _apply(self, fn):
        """
        Applies a function recursively to all tensors that are sub-modules or
        direct attributes of this module.

        This method overrides the `_apply` method of `nn.Module` to ensure that
        the function `fn` is also applied to the `radial` attribute, which is
        assumed to be a `nn.Module` as well. This is necessary to propagate
        changes like moving tensors to a different device
        (e.g., CPU to GPU) or changing their data type.
        """

        super(TopologicalMap, self)._apply(fn)
        self.radial = self.radial._apply(fn)
        return self

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

    def get_representation(self, x, rtype="point", neighborhood_std=None):
        """
        Generates the representation of the Best Matching Unit (BMU) based on
        the specified type requested.

        Args:
            - x (torch.Tensor): Input tensor.
            - rtype (str, optional): Type of the representation. Supported
              types are 'point' (default) and 'grid'.
            - neighborhood_std (float, optional): Standard deviation for the
              neighborhood function. Defaults to current neighborhood_std.

        Returns:
            torch.Tensor or None: BMU representation based on the specified
            type; None if BMU is unavailable.
        """

        self.bmu = self.find_bmu(x)
        if self.bmu is not None:
            if rtype == "point":
                if self.output_dims == 1:
                    return self.bmu.float()

                elif self.output_dims == 2:
                    row = self.bmu // self.side
                    col = self.bmu % self.side
                    return torch.stack([row, col]).T.float()

            elif rtype == "grid":
                if neighborhood_std is None:
                    neighborhood_std = self.curr_neighborhood_std
                phi = self.radial(self.bmu, neighborhood_std)
                return phi
        else:
            return None

    def backward(self, point, neighborhood_std=None):
        """
        Executes the backward pass for a specified point.

        Args:
            - point (int, int): Target point for the backward pass.
            - neighborhood_std (float, optional): Standard deviation for the
              radial basis function. Defaults to current neighborhood_std.

        Returns:
            torch.Tensor: Result of the backward pass for the specified point.
        """

        if neighborhood_std is None:
            neighborhood_std = self.curr_neighborhood_std
        phi = self.radial(point, neighborhood_std, as_point=True)
        max_idx = phi.argmax()
        hov_phi = torch.nn.functional.one_hot(
            max_idx, self.output_size
        ).float()
        output = torch.matmul(hov_phi, self.weights.T)
        return output


class LossFactory:
    """
    This class is responsible for building the loss functon for a SOM or STM
    model.

    Parameters:
        - model (torch.nn.Module): The SOM or STM model to update.
        - mode (str): Specifies the update type, either 'som' or 'stm'.
        - kernel_function (callable, optional): Defines the kernel
          function.  Defaults to:
                - <lambda phi: phi> if mode is 'som'.
                - <lambda phi, psi: phi * psi> if mode is 'stm'.
    """

    def __init__(
        self,
        model,
        mode="som",
        kernel_function=None,
    ):
        self.model = model
        self.mode = mode
        if kernel_function is None:
            if self.mode == "som":
                self.kernel_function = lambda phi: phi
            elif self.mode == "stm":
                self.kernel_function = lambda phi, psi: phi * psi
        else:
            self.kernel_function = kernel_function

    def losses(
        self,
        norms2,
        neighborhood_std,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        """
        Compute the SOM/STM loss.

        Parameters:
            - norms2 (array-like): The squared norm of some input data.
            - neighborhood_std (float): The standard deviation for neighborhood
              radial calculation.
            - anchors (array-like, optional): Labels or anchors for
              neighborhood modulation. Default is None.
            - neighborhood_std_anchors (float, optional): The standard
              deviation for anchors  neighborhood modulation. Default is
              neighborhood_std.

        Returns:
            array-like: The values of the computed losses.
        """

        self.model.curr_neighborhood_std = neighborhood_std
        self.model.bmu = self.model.find_bmu(norms2)

        # If anchors are not provided, calculate loss without anchors
        if anchors is None:
            phi = self.model.radial(self.model.bmu, neighborhood_std)
            _losses = 0.5 * norms2 * self.kernel_function(phi)
        # If anchors are provided, incorporate them into the loss calculation
        else:
            if neighborhood_std_anchors is None:
                neighborhood_std_anchors = neighborhood_std
            phi = self.model.radial(self.model.bmu, neighborhood_std)
            psi = self.model.radial(
                anchors, neighborhood_std_anchors, as_point=True
            )
            _losses = 0.5 * norms2 * self.kernel_function(phi, psi)

        return _losses


class LossEfficacyFactory(LossFactory):
    def __init__(self, efficacy_radial_sigma, efficacy_decay, *args, **kwargs):
        super(LossEfficacyFactory, self).__init__(*args, **kwargs)
        self.efficacy_radial_sigma = efficacy_radial_sigma
        self.efficacy_decay = efficacy_decay
        self._efficacies = torch.zeros(self.model.output_size)
        self._inefficacies = 1.0 - torch.zeros(self.model.output_size)

    def loss(
        self,
        norms2,
        neighborhood_baseline,
        neighborhood_max,
        anchors=None,
        neighborhood_std_anchors=None,
    ):

        _neighborhood_std = neighborhood_baseline + self._inefficacies * (
            neighborhood_max - neighborhood_baseline
        )

        losses = self.losses(
            norms2,
            _neighborhood_std,
            anchors,
            neighborhood_std_anchors,
        )

        _loss = losses * self._inefficacies.reshape(-1, 1)

        norm_radial_bases = torch.exp(
            -0.5 * (self.efficacy_radial_sigma**-2) * norms2
        )
        self._efficacies = self._efficacies + self.decay * (
            norm_radial_bases - self._efficacies
        )
        self._inefficacies = 1.0 - self._efficacies

        return _loss.mean()


class Updater(LossFactory):
    """
    This class is responsible for updating a SOM or STM model.

    Parameters:
        - model (torch.nn.Module): The SOM or STM model to update.
        - learning_rate (float): The optimizer's learning rate.
        - mode (str): Specifies the update type, either 'som' or 'stm'.
        - kernel_function (callable, optional): Defines the kernel
          function.  Defaults to:
                - <lambda phi: phi> if mode is 'som'.
                - <lambda phi, psi: phi * psi> if mode is 'stm'.
    """

    def __init__(self, model, learning_rate, mode="som", kernel_function=None):

        super().__init__(model, mode, kernel_function)

        self.optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate
        )

    def __call__(
        self,
        output,
        neighborhood_std,
        learning_modulation,
        anchors=None,
        neighborhood_std_anchors=None,
    ):
        if self.mode == "som":
            losses = self.loss(output, neighborhood_std)
        elif self.mode == "stm":
            losses = self.losses(
                output, neighborhood_std, anchors, neighborhood_std_anchors
            )
        else:
            raise ValueError("Invalid mode. Use 'som' or 'stm'.")

        loss = (learning_modulation * losses).mean()
        unmodulated_loss = losses.mean()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss, unmodulated_loss
