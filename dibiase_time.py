from .xsrp import XSrp

from .grids import (
    UniformAngularGrid, UniformCartesianGrid
)


class DibiaseTime(XSrp):
    def __init__(self, fs: float, mode, n_grid_cells,
                 mic_positions=None, room_dims=None, c=343):
        super().__init__(fs, mic_positions, room_dims, c)

        if mode not in ["2D", "3D", "doa_1D", "doa_2D"]:
            raise ValueError("mode must be one of '2D', '3D', 'doa_1D', 'doa_2D'")
        
        self.mode = mode

        self.n_grid_cells = n_grid_cells
    
    def create_initial_candidate_grid(self, room_dims):
        if self.mode in ["2D", "3D"]:
            return UniformCartesianGrid(room_dims, self.n_grid_cells)
        elif self.mode == "doa_1D":
            return UniformAngularGrid(self.n_grid_cells)
        elif self.mode == "doa_2D":
            return UniformAngularGrid(self.n_grid_cells, self.n_grid_cells)


    def create_time_space_mapper(self, mic_positions, candidate_grid):
        