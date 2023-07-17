import numpy as np

from .grids import Grid


def tdoa_mapper(grid: Grid, microphone_positions: np.array):
    """Create a TDOA mapper for the given grid.

    A TDOA mapper associates a theoretical Time Difference
    of Arrival (TDOA) value
    for each grid cell and microphone pair.

    Parameters
    ----------
    grid : Grid
        The grid of candidate positions to create the temporal-spatial mapper for.
        The grid is of shape 2D (n_points, 2/3)
    
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.
        
    Returns
    -------
    mapper : np.array (n_candidate_positions, n_microphones, n_microphones) 
        The tdoa map.
    """

    grid = grid.asarray()

    if grid.ndim != 2:
        raise ValueError("The grid must be 2D.")

    if grid.shape[-1] != microphone_positions.shape[-1]:
        raise ValueError("The grid and microphone positions must have the same number of dimensions.")

    n_candidate_positions, n_dims = grid.shape

    n_microphones = microphone_positions.shape[0]

    mapper = np.zeros((n_candidate_positions, n_microphones, n_microphones))

    for i in range(n_microphones):
        for j in range(n_microphones):
            if i != j:
                mapper[:, i, j] = _compute_theoretical_tdoa(grid, microphone_positions[i], microphone_positions[j])
    return mapper


def fractional_sample_mapper(grid: Grid, microphone_positions: np.array, sr: float):
    """Create a fractional sample mapper for the given grid.

    This map associates a fractional sample value for each grid cell and microphone pair.
    The fractional sample value is computed as the theoretical TDOA multiplied by the sampling rate
    of the signals.

    Parameters
    ----------
    grid : Grid
        The grid of candidate positions to create the temporal-spatial mapper for.
        The grid is of shape 2D (n_points, 2/3)
    
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.
        
    sr : float
        The sampling rate of the signals.

    Returns
    -------
    mapper : np.array (n_candidate_positions, n_microphones, n_microphones) 
    """

    return tdoa_mapper(grid, microphone_positions) * sr


def integer_sample_mapper(grid: Grid, microphone_positions: np.array, sr: float):
    """Create an integer sample mapper for the given grid.

    This map associates an integer sample value for each grid cell and microphone pair.
    The integer sample value is computed as the theoretical TDOA multiplied by the sampling rate
    of the signals and rounded to the nearest integer.

    Parameters
    ----------
    grid : Grid
        The grid of candidate positions to create the temporal-spatial mapper for.
        The grid is of shape 2D (n_points, 2/3)
    
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.
        
    sr : float
        The sampling rate of the signals.

    Returns
    -------
    mapper : np.array (n_candidate_positions, n_microphones, n_microphones) 
    """

    fractional_mapper = fractional_sample_mapper(grid, microphone_positions, sr)
    integer_mapper = np.ceil(fractional_mapper).astype(int)
    
    return integer_mapper


def _compute_theoretical_tdoa(candidate_grid, mic_0, mic_1, c=343.0):
    """Compute the theoretical TDOA between a grid of candidate positions and a microphone pair.
    The TDOA is computed as the difference between the Time of Flight (TOF) of the candidate position
    to the first microphone and the TOF of the candidate position to the second microphone.
    

    Parameters
    ----------
    candidate_grid : np.array (n_points, n_dimensions)
        The candidate position.
    mic_0 : np.array (n_dimensions)
        The first microphone.
    mic_1 : np.array (n_dimensions)
        The second microphone.
    c : float
        The speed of sound. Defaults to 343.0 m/s.
    
    Returns
    -------
    tdoa : float
        The theoretical TDOA between the candidate grid and the microphone pair.
    """

    tof_0 = np.linalg.norm(candidate_grid - mic_0, axis=-1)/c
    tof_1 = np.linalg.norm(candidate_grid - mic_1, axis=-1)/c
    tdoa = tof_0 - tof_1

    return tdoa