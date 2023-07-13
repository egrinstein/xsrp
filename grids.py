import numpy as np


class GridCell:
    def __init__(self, position, likelihood=None):
        self.position = position
        self.likelihood = likelihood

    def set_likelihood(self, value):
        self.likelihood = value
    

class UniformCartesianGrid:
    """Create an uniform cartesian grid in 2D or 3D for a cuboid shaped room
    """

    def __init__(self, bounds: list[float], n_grid_cells_per_dim: int): 
        """
        bounds: List of floats, each float representing the length of the cuboid shape in that dimension
        n_grid_cells_per_dim: number of grid cells per dimension
        """

        self.bounds = np.array(bounds)
        self.n_grid_cells_per_dim = n_grid_cells_per_dim

        self._create_grid()
    
    def _create_grid(self):
        self.grid_shape = [self.n_grid_cells_per_dim] * len(self.bounds) + [len(self.bounds)]

        self.positions = np.zeros(self.grid_shape)
        self.likelihoods = np.ones(self.grid_shape[:-1]) / np.prod(self.grid_shape[:-1])

        cell_resolution = self.bounds / (self.n_grid_cells_per_dim)
        start_position = cell_resolution / 2

        xrange = np.linspace(start_position[0], self.bounds[0] - start_position[0], self.n_grid_cells_per_dim)
        yrange = np.linspace(start_position[1], self.bounds[1] - start_position[1], self.n_grid_cells_per_dim)
        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                if len(self.bounds) == 2: # 2D
                    self.positions[i, j] = np.array([x, y])
                elif len(self.bounds) == 3: # 3D
                    zrange = np.linspace(start_position[2], self.bounds[2] - start_position[2], self.n_grid_cells_per_dim)
                    for k, z in enumerate(zrange):
                        self.positions[i, j, k] = np.array([x, y, z])
    
    def set_likelihoods(self, likelihoods):
        self.likelihoods = likelihoods

    def __iter__(self):
        flattened_positions = self.positions.reshape(-1, len(self.bounds))
        flattened_likelihoods = self.likelihoods.flatten()

        for position, likelihood in zip(flattened_positions, flattened_likelihoods):
            yield GridCell(position, likelihood)

    def __getitem__(self, key):
        position = self.positions[key]
        likelihood = self.likelihoods[key]
        return GridCell(position, likelihood)


class UniformAngularGrid:
    def __init__(self, n_azimuth_cells, n_elevation_cells=0):
        self.n_azimuth_cells = n_azimuth_cells
        self.n_elevation_cells = n_elevation_cells

        self._create_grid()
    
    def _create_grid(self):

        if self.n_elevation_cells == 0:
            self.grid_shape = np.array([self.n_azimuth_cells, 2])
        else:
            self.grid_shape = np.array([self.n_azimuth_cells, self.n_elevation_cells, 3])

        self.positions = np.zeros(self.grid_shape)
        self.likelihoods = np.ones(self.grid_shape[:-1]) / np.prod(self.grid_shape[:-1])

        azimuth_range = np.linspace(0, 2 * np.pi, self.n_azimuth_cells, endpoint=False)
        if self.n_elevation_cells == 0:
            for i, azimuth in enumerate(azimuth_range):
                self.positions[i] = np.array([np.cos(azimuth), np.sin(azimuth)])
        else:
            elevation_range = np.linspace(0, np.pi, self.n_elevation_cells, endpoint=False)
            for i, azimuth in enumerate(azimuth_range):
                for j, elevation in enumerate(elevation_range):
                    self.positions[i, j] = np.array([np.cos(azimuth) * np.sin(elevation), np.sin(azimuth) * np.sin(elevation), np.cos(elevation)])

    def set_likelihoods(self, likelihoods):
        self.likelihoods = likelihoods

    def __getitem__(self, key):
        position = self.positions[key]
        likelihood = self.likelihoods[key]
        return GridCell(position, likelihood)

    def __iter__(self):
        flattened_positions = self.positions.reshape(-1, self.positions.shape[-1])
        flattened_likelihoods = self.likelihoods.flatten()

        for position, likelihood in zip(flattened_positions, flattened_likelihoods):
            yield GridCell(position, likelihood)
