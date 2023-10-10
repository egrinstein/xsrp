from typing import Any
import numpy as np


class Grid:
    def __init__(self, positions: np.array):
        self.positions = positions

    def __iter__(self):
        flattened_positions = self.positions.reshape(-1, self.positions.shape[-1])

        for position in flattened_positions:
            yield position

    def __getitem__(self, key):
        position = self.positions[key]
        return position
    
    def __setitem__(self, key, value):
        self.positions[key] = value

    def asarray(self):
        return self.positions.reshape(-1, self.positions.shape[-1])

    def __len__(self):
        return np.prod(self.positions.shape[:-1])


class UniformCartesianGrid(Grid):
    """Create an uniform cartesian grid in 2D or 3D for a cuboid shaped room
    """

    def __init__(self, bounds: list[float], n_grid_cells_per_dim: int): 
        """
        bounds: List of floats, each float representing the length of the cuboid shape in that dimension
        n_grid_cells_per_dim: number of grid cells per dimension
        """

        self.bounds = np.array(bounds)
        self.n_grid_cells_per_dim = n_grid_cells_per_dim

        positions = self._create_grid()

        super().__init__(positions)

    def _create_grid(self):
        self.grid_shape = [self.n_grid_cells_per_dim] * len(self.bounds) + [len(self.bounds)]

        positions = np.zeros(self.grid_shape)
     
        cell_resolution = self.bounds / (self.n_grid_cells_per_dim)
        start_position = cell_resolution / 2

        xrange = np.linspace(start_position[0], self.bounds[0] - start_position[0], self.n_grid_cells_per_dim)
        yrange = np.linspace(start_position[1], self.bounds[1] - start_position[1], self.n_grid_cells_per_dim)
        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                if len(self.bounds) == 2: # 2D
                    positions[i, j] = np.array([x, y])
                elif len(self.bounds) == 3: # 3D
                    zrange = np.linspace(start_position[2], self.bounds[2] - start_position[2], self.n_grid_cells_per_dim)
                    for k, z in enumerate(zrange):
                        positions[i, j, k] = np.array([x, y, z])
    
        return positions

class UniformSphericalGrid(Grid):
    def __init__(self, n_azimuth_cells, n_elevation_cells=0):
        self.n_azimuth_cells = n_azimuth_cells
        self.n_elevation_cells = n_elevation_cells

        positions = self._create_grid()

        super().__init__(positions)

    def _create_grid(self):

        if self.n_elevation_cells == 0:
            self.grid_shape = np.array([self.n_azimuth_cells, 2])
        else:
            self.grid_shape = np.array([self.n_azimuth_cells, self.n_elevation_cells, 3])

        positions = np.zeros(self.grid_shape)

        self.azimuth_range = np.linspace(0, 2 * np.pi, self.n_azimuth_cells, endpoint=False)
        if self.n_elevation_cells == 0:
            for i, azimuth in enumerate(self.azimuth_range):
                positions[i] = np.array([np.cos(azimuth), np.sin(azimuth)])
        else:
            self.elevation_range = np.linspace(0, np.pi, self.n_elevation_cells, endpoint=False)
            for i, azimuth in enumerate(self.azimuth_range):
                for j, elevation in enumerate(self.elevation_range):
                    positions[i, j] = np.array([np.cos(azimuth) * np.sin(elevation), np.sin(azimuth) * np.sin(elevation), np.cos(elevation)])

        return positions
