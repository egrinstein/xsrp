import numpy as np

from .xsrp import XSrp

from .grids import (
    UniformSphericalGrid, UniformCartesianGrid
)
from .spatial_mappers import (
    integer_sample_mapper
)
from .signal_features.cross_correlation import (
    cross_correlation
)
from .projectors import (
    integer_sample_projector
)
from .grid_search import (
    argmax_grid_searcher
)


class DiBiaseTimeSrp(XSrp):
    def __init__(self, fs: float, mode, n_grid_cells,
                 mic_positions=None, room_dims=None, c=343):
        if mode not in ["2D", "3D", "doa_1D", "doa_2D"]:
            raise ValueError("mode must be one of '2D', '3D', 'doa_1D', 'doa_2D'")
        
        self.mode = mode

        self.n_grid_cells = n_grid_cells

        super().__init__(fs, mic_positions, room_dims, c)

    
    def create_initial_candidate_grid(self, room_dims):
        if self.mode in ["2D", "3D"]:
            return UniformCartesianGrid(room_dims, self.n_grid_cells)
        elif self.mode == "doa_1D":
            return UniformSphericalGrid(self.n_grid_cells)
        elif self.mode == "doa_2D":
            return UniformSphericalGrid(self.n_grid_cells, self.n_grid_cells)


    def create_spatial_mapper(self, mic_positions, candidate_grid):
        return integer_sample_mapper(candidate_grid, mic_positions, self.fs)

    def compute_signal_features(self, mic_signals):
        return cross_correlation(mic_signals, abs=True, return_lags=False)

    def project_features(self,
                         candidate_grid,
                         spatial_mapper,
                         signal_features):
        return integer_sample_projector(candidate_grid, spatial_mapper, signal_features)

    def grid_search(self, candidate_grid, srp_map, estimated_positions):
        estimated_positions = argmax_grid_searcher(candidate_grid, srp_map)
        
        # An empty grid is returned to indicate that the algorithm has converged
        return estimated_positions, np.array([])