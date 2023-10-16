import numpy as np

from .xsrp import XSrp

from .grids import (
    Grid, UniformSphericalGrid, UniformCartesianGrid
)
from .signal_features.cross_correlation import cross_correlation
from .signal_features.gcc_phat import gcc_phat

from .srp_mappers import (
    temporal_projector,
    frequency_projector
)
from .grid_search import (
    argmax_grid_searcher
)


class ConventionalSrp(XSrp):
    """Conventional SRP algorithm as described by DiBiase et al. (2001).
    
    Parameters
    ----------
    fs : float
        The sampling rate of the signals.
    grid_type : str
        The type of grid used in the algorithm. Must be one of '2D', '3D', 'doa_1D', 'doa_2D'.
    n_grid_cells : int
        The number of grid cells to use in each dimension.
    mic_positions : np.array (n_mics, n_dimensions), optional
        The positions of the microphones. If None, the positions must be
        specified when calling the forward method. Defaults to None.
    room_dims : np.array (n_dimensions,), optional
        The dimensions of the room. If None, the dimensions must be 
        specified when calling the forward method. Defaults to None.
    c : float, optional
        The speed of sound. Defaults to 343.
    mode : str, optional
        The mode of the algorithm. Must be one of 'gcc_phat_freq', 'gcc_phat_time', 'cross_correlation'.
        Defaults to 'gcc_phat_time'.
    interpolation : bool, optional
        Whether to use fractional sample interpolation. Defaults to False.
    n_average_samples : int, optional
        The number of cross-correlation samples to average over. Defaults to 1.
        
    """
    
    def __init__(self, fs: float, grid_type, n_grid_cells,
                 mic_positions=None, room_dims=None, c=343,
                 mode="gcc_phat_time",
                 interpolation=False,
                 n_average_samples=1,
                 n_dft_bins=1024,
                 freq_cutoff_in_hz=None):
        if grid_type not in ["2D", "3D", "doa_1D", "doa_2D"]:
            raise ValueError("grid_type must be one of '2D', '3D', 'doa_1D', 'doa_2D'")
        
        if mode not in ["gcc_phat_freq", "gcc_phat_time", "cross_correlation"]:
            raise ValueError(
                "mode must be one of {}".format(
                    ["gcc_phat_freq", "gcc_phat_time", "cross_correlation"]
                )
            )
    
        freq_cutoff_in_hz = freq_cutoff_in_hz or fs//2
        
        self.grid_type = grid_type
        self.n_grid_cells = n_grid_cells
        
        self.mode = mode
        self.interpolation = interpolation
        self.n_average_samples = n_average_samples
        self.n_dft_bins = n_dft_bins
        self.freq_cutoff = freq_cutoff_in_hz*n_dft_bins//fs

        super().__init__(fs, mic_positions, room_dims, c)
    
    def create_initial_candidate_grid(self, room_dims):
        if self.grid_type in ["2D", "3D"]:
            return UniformCartesianGrid(room_dims, self.n_grid_cells)
        elif self.grid_type == "doa_1D":
            return UniformSphericalGrid(self.n_grid_cells)
        elif self.grid_type == "doa_2D":
            return UniformSphericalGrid(self.n_grid_cells, self.n_grid_cells)

    def compute_signal_features(self, mic_signals):
        if self.mode == "cross_correlation":
            return cross_correlation(mic_signals, abs=True, return_lags=False)
        elif self.mode == "gcc_phat_time":
            return gcc_phat(mic_signals, abs=True, return_lags=False, ifft=True)
        elif self.mode == "gcc_phat_freq":
            return gcc_phat(mic_signals, abs=True, return_lags=False, ifft=False,
                            n_dft_bins=self.n_dft_bins)

    def create_srp_map(self,
                       mic_positions: np.array,
                       candidate_grid: Grid,
                       signal_features: np.array):

        if self.mode == "gcc_phat_freq":
            return frequency_projector(
                mic_positions,
                candidate_grid,
                signal_features,
                self.fs,
                freq_cutoff=self.freq_cutoff)
        else:
            return temporal_projector(
                mic_positions,
                candidate_grid,
                signal_features,
                self.fs,
                n_average_samples=self.n_average_samples,
                interpolate=self.interpolation)
            
    def grid_search(self, candidate_grid, srp_map, estimated_positions):
        estimated_positions = argmax_grid_searcher(candidate_grid, srp_map)
        
        # An empty grid is returned to indicate that the algorithm has converged
        return estimated_positions, np.array([])
