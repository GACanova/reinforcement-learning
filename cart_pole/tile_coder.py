"""
Tile Coder for State Space Representation in Reinforcement Learning.
"""

import numpy as np

from config import NUM_TILINGS, TILES_PER_DIMENSION, LOWER_BOUNDS, UPPER_BOUNDS


class TileCoder:
    """
    Tile coding technique implementation for discretizing continuous state spaces.

    This class encodes multi-dimensional continuous state spaces using a tile coding system.
    It creates multiple overlapping tilings to provide a more fine-grained discretization
    without dramatically increasing the dimensionality. This representation is useful in
    reinforcement learning for function approximation where the state space is continuous.

    Attributes:
        num_tilings (int): The number of tilings used in the coding.
        tiles_per_dimension (numpy.ndarray): The number of tiles for each state dimension.
        lower_bounds (numpy.ndarray): The lower bound of the state space for each dimension.
        upper_bounds (numpy.ndarray): The upper bound of the state space for each dimension.
        dim (int): The number of dimensions in the state space.
        tile_sizes (numpy.ndarray): The size of each tile along each dimension.
        number_of_features (int): The total number of distinct features (tiles) in the coding.
        offsets (numpy.ndarray): The offsets for each tiling, used to create overlapping tilings.

    Methods:
        get_tile_indices(state): Computes the tile indices in the feature space for a given state.

    Parameters:
        num_tilings (int): See attributes.
        tiles_per_dimension (tuple of int): See attributes.
        lower_bounds (tuple): See attributes.
        upper_bounds (tuple): See attributes.
        offsets (list of tuples, optional): Custom-defined offsets for the tilings.

    Raises:
        AssertionError: If there is a mismatch in the dimensionality of input parameters.
    """

    def __init__(
        self,
        num_tilings=NUM_TILINGS,
        tiles_per_dimension=TILES_PER_DIMENSION,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        offsets=None,
    ):
        assert (
            len(tiles_per_dimension) == len(lower_bounds) == len(upper_bounds)
        ), "Dimension mismatch input parameters."

        self.num_tilings = num_tilings
        self.tiles_per_dimension = np.array(tiles_per_dimension, dtype=np.int32)
        self.lower_bounds = np.array(lower_bounds)
        self.upper_bounds = np.array(upper_bounds)
        self.dim = len(tiles_per_dimension)
        self.tile_sizes = (self.upper_bounds - self.lower_bounds) / (
            self.tiles_per_dimension
        )
        self.number_of_features = self.num_tilings * np.prod(self.tiles_per_dimension)

        if offsets is None:
            self.offsets = [
                tuple((2 * i - 1) * k for i in range(1, self.dim + 1))
                for k in range(self.num_tilings)
            ]
        else:
            assert len(offsets) == self.dim, "Dimension mismatch offsets."
            self.offsets = offsets

        self.offsets = np.array(self.offsets) / self.num_tilings

    def get_tile_indices(self, state):
        tile_indices = []

        for tiling, offset in enumerate(self.offsets):
            adjusted_state = (state - self.lower_bounds) / self.tile_sizes + offset
            indices = np.floor(adjusted_state).astype(int) % self.tiles_per_dimension
            tile_indices.append((tiling, *indices))

        return tile_indices
