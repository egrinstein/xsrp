from .xsrp import XSrp

from .grids import (
    create_uniform_grid_2d, create_uniform_grid_3d,
    create_grid_doa_1d, create_grid_doa_2d
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
        if self.mode == "2D":
            return create_uniform_grid_2d(room_dims)
        elif self.mode == "3D":
            return create_uniform_grid_3d(room_dims)
        elif self.mode == "doa_1D":
            return create_grid_doa_1d(room_dims)
        elif self.mode == "doa_2D":
            return create_grid_doa_2d(room_dims)
