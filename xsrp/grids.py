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
    def __init__(self, n_azimuth_cells, n_elevation_cells=0, 
                 azimuth_min=None, azimuth_max=None,
                 elevation_min=None, elevation_max=None):
        """
        Create a uniform spherical grid for DOA estimation.
        
        Parameters
        ----------
        n_azimuth_cells : int
            Number of azimuth cells in the grid.
        n_elevation_cells : int, optional
            Number of elevation cells. If 0, creates a 1D DOA grid (azimuth only).
            Defaults to 0.
        azimuth_min : float, optional
            Minimum azimuth angle in radians. If None, defaults to 0.
        azimuth_max : float, optional
            Maximum azimuth angle in radians. If None, defaults to 2*pi.
        elevation_min : float, optional
            Minimum elevation angle in radians. If None, defaults to 0.
            Only used when n_elevation_cells > 0.
        elevation_max : float, optional
            Maximum elevation angle in radians. If None, defaults to pi.
            Only used when n_elevation_cells > 0.
        """
        self.n_azimuth_cells = n_azimuth_cells
        self.n_elevation_cells = n_elevation_cells
        
        # Set defaults for azimuth range
        self.azimuth_min = azimuth_min if azimuth_min is not None else 0.0
        self.azimuth_max = azimuth_max if azimuth_max is not None else 2 * np.pi
        
        # Set defaults for elevation range
        self.elevation_min = elevation_min if elevation_min is not None else 0.0
        self.elevation_max = elevation_max if elevation_max is not None else np.pi
        
        # Validate ranges
        if self.azimuth_min >= self.azimuth_max:
            raise ValueError(f"azimuth_min ({self.azimuth_min}) must be less than azimuth_max ({self.azimuth_max})")
        if self.azimuth_min < 0 or self.azimuth_max > 2 * np.pi:
            raise ValueError(f"Azimuth range must be within [0, 2*pi]. Got [{self.azimuth_min}, {self.azimuth_max}]")
        
        if self.n_elevation_cells > 0:
            if self.elevation_min >= self.elevation_max:
                raise ValueError(f"elevation_min ({self.elevation_min}) must be less than elevation_max ({self.elevation_max})")
            if self.elevation_min < 0 or self.elevation_max > np.pi:
                raise ValueError(f"Elevation range must be within [0, pi]. Got [{self.elevation_min}, {self.elevation_max}]")

        positions = self._create_grid()

        super().__init__(positions)

    def _create_grid(self):
        # Generate full azimuth range, then filter
        full_azimuth_range = np.linspace(0, 2 * np.pi, self.n_azimuth_cells, endpoint=False)
        # Filter azimuth range based on min/max
        azimuth_mask = (full_azimuth_range >= self.azimuth_min) & (full_azimuth_range < self.azimuth_max)
        self.azimuth_range = full_azimuth_range[azimuth_mask]
        n_azimuth_filtered = len(self.azimuth_range)
        
        if n_azimuth_filtered == 0:
            raise ValueError(f"No azimuth cells in the specified range [{self.azimuth_min}, {self.azimuth_max})")
        
        if self.n_elevation_cells == 0:
            # 1D DOA grid (azimuth only)
            self.grid_shape = np.array([n_azimuth_filtered, 2])
            positions = np.zeros(self.grid_shape)
            for i, azimuth in enumerate(self.azimuth_range):
                positions[i] = np.array([np.cos(azimuth), np.sin(azimuth)])
        else:
            # 2D DOA grid (azimuth + elevation)
            # Generate full elevation range, then filter
            full_elevation_range = np.linspace(0, np.pi, self.n_elevation_cells, endpoint=False)
            # Filter elevation range based on min/max
            elevation_mask = (full_elevation_range >= self.elevation_min) & (full_elevation_range < self.elevation_max)
            self.elevation_range = full_elevation_range[elevation_mask]
            n_elevation_filtered = len(self.elevation_range)
            
            if n_elevation_filtered == 0:
                raise ValueError(f"No elevation cells in the specified range [{self.elevation_min}, {self.elevation_max})")
            
            self.grid_shape = np.array([n_azimuth_filtered, n_elevation_filtered, 3])
            positions = np.zeros(self.grid_shape)
            for i, azimuth in enumerate(self.azimuth_range):
                for j, elevation in enumerate(self.elevation_range):
                    positions[i, j] = np.array([
                        np.cos(azimuth) * np.sin(elevation),
                        np.sin(azimuth) * np.sin(elevation),
                        np.cos(elevation)
                    ])

        return positions
