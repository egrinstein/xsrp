import numpy as np

from .xsrp import XSrp

from .grids import (
    Grid, UniformSphericalGrid, UniformCartesianGrid
)
from .signal_features.cross_correlation import cross_correlation
from .signal_features.gcc_phat import gcc_phat
from .signal_features.frequency_weighting import compute_frequency_weights

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
    sharpening : float, optional
        The exponent to raise the SRP map to. Defaults to 1.
    frequency_weighting : str, optional
        Frequency weighting method: None, 'coherence', 'sparsity', or 'par'.
        Only applies when mode='gcc_phat_freq'. Defaults to None.
        
    """
    
    def __init__(self, fs: float, grid_type, n_grid_cells,
                 mic_positions=None, room_dims=None, c=343,
                 mode="gcc_phat_time",
                 interpolation=False,
                 n_average_samples=1,
                 n_dft_bins=1024,
                 freq_cutoff_in_hz=None,
                 sharpening=1.0,
                 frequency_weighting=None):
        if grid_type not in ["2D", "3D", "doa_1D", "doa_2D"]:
            raise ValueError("grid_type must be one of '2D', '3D', 'doa_1D', 'doa_2D'")
        
        available_modes = ["gcc_phat_freq", "gcc_phat_time", "cross_correlation"]
        if mode not in available_modes:
            raise ValueError(
                "mode must be one of {}".format(available_modes)
            )
    
        freq_cutoff_in_hz = freq_cutoff_in_hz or fs//2
        
        self.grid_type = grid_type
        self.n_grid_cells = n_grid_cells
        
        self.mode = mode
        self.interpolation = interpolation
        self.n_average_samples = n_average_samples
        self.n_dft_bins = n_dft_bins
        self.freq_cutoff = freq_cutoff_in_hz*n_dft_bins//fs
        self.sharpening = sharpening
        self.frequency_weighting = frequency_weighting
        
        # Validate frequency_weighting parameter
        if frequency_weighting is not None:
            if mode != "gcc_phat_freq":
                raise ValueError("frequency_weighting only applies when mode='gcc_phat_freq'")
            if frequency_weighting not in [None, 'coherence', 'sparsity', 'par']:
                raise ValueError(
                    f"frequency_weighting must be one of None, 'coherence', 'sparsity', 'par'. "
                    f"Got: {frequency_weighting}"
                )

        super().__init__(fs, mic_positions, room_dims, c)
        
        # Store frequency-domain signals when needed for coherence weighting
        self._mic_signals_dft = None
    
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
            # Store frequency-domain signals for coherence weighting if needed
            if self.frequency_weighting == 'coherence':
                self._mic_signals_dft = np.fft.rfft(mic_signals, n=self.n_dft_bins)
            return gcc_phat(mic_signals, abs=True, return_lags=False, ifft=False,
                            n_dft_bins=self.n_dft_bins)

    def create_srp_map(self,
                       mic_positions: np.array,
                       candidate_grid: Grid,
                       signal_features: np.array):

        if self.mode == "gcc_phat_freq":
            # Compute frequency weights if weighting is enabled
            frequency_weights = None
            if self.frequency_weighting is not None:
                # First, compute per-frequency SRP maps (without weights) to compute weights
                # Use return_per_freq to get per-frequency maps efficiently
                _, per_freq_srp_maps = frequency_projector(
                    mic_positions,
                    candidate_grid,
                    signal_features,
                    self.fs,
                    freq_cutoff=self.freq_cutoff,
                    frequency_weights=None,  # No weights yet
                    return_per_freq=True
                )
                
                # Truncate mic_signals_dft to match freq_cutoff if needed
                mic_signals_dft_truncated = None
                if self._mic_signals_dft is not None:
                    # freq_cutoff is the number of bins to keep (0 to freq_cutoff-1)
                    # _mic_signals_dft has shape (n_mics, n_dft_bins//2 + 1)
                    # We need to truncate to [:freq_cutoff] if freq_cutoff is less than full size
                    n_full_bins = self._mic_signals_dft.shape[1]
                    if self.freq_cutoff is not None and self.freq_cutoff < n_full_bins:
                        mic_signals_dft_truncated = self._mic_signals_dft[:, :self.freq_cutoff]
                    else:
                        mic_signals_dft_truncated = self._mic_signals_dft
                
                # Compute weights based on the method
                frequency_weights = compute_frequency_weights(
                    per_freq_srp_maps,
                    mic_signals_dft=mic_signals_dft_truncated,
                    method=self.frequency_weighting
                )
            
            srp_map = frequency_projector(
                mic_positions,
                candidate_grid,
                signal_features,
                self.fs,
                freq_cutoff=self.freq_cutoff,
                frequency_weights=frequency_weights)
        else:
            srp_map = temporal_projector(
                mic_positions,
                candidate_grid,
                signal_features,
                self.fs,
                n_average_samples=self.n_average_samples,
                interpolate=self.interpolation)
        
        if self.sharpening != 1.0:
            srp_map[srp_map < 0] = 0
            srp_map = srp_map ** self.sharpening
            
        return srp_map
            
    def grid_search(self, candidate_grid, srp_map, estimated_positions, signal_features):
        estimated_positions = argmax_grid_searcher(candidate_grid, srp_map)
        
        # An empty grid is returned to indicate that the algorithm has converged
        return estimated_positions, np.array([]), signal_features
