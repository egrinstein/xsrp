"""
This file contains functions to create spatial mappers, which are used to
project spatial information into a temporal domain. These mappers are
the bridge between the candidate positions and the microphone signals.
"""

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

    # Detect if this is a DOA grid (unit vectors) vs positional grid
    # DOA grids have positions with norm approximately 1.0
    grid_norms = np.linalg.norm(grid, axis=-1)
    is_doa_grid = np.allclose(grid_norms, 1.0, atol=0.01)

    n_candidate_positions, n_dims = grid.shape

    n_microphones = microphone_positions.shape[0]

    mapper = np.zeros((n_candidate_positions, n_microphones, n_microphones))

    for i in range(n_microphones):
        for j in range(n_microphones):
            if i != j:
                mapper[:, i, j] = _compute_theoretical_tdoa(
                    grid, microphone_positions[i], microphone_positions[j], 
                    far_field=is_doa_grid
                )
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
    # Use round instead of ceil for nearest neighbor interpolation
    # This reduces quantization error from up to 1 sample to ~0.5 samples
    integer_mapper = np.round(fractional_mapper).astype(int)
    
    return integer_mapper


def frequency_delay_mapper(candidate_grid, microphone_positions: np.array, freqs: np.array):
    """Create a frequency delay mapper for the given grid.

    This map associates a a complex value exp(j*f*tau) for each grid cell and microphone pair, where
    f is a frequency band being analyzed and tau is the theoretical TDOA between the grid cell and the
    microphone pair.

    Parameters
    ----------
    grid : Grid
        The grid of candidate positions to create the temporal-spatial mapper for.
        The grid is of shape 2D (n_points, 2/3)
    
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.    
    
    freqs : np.array (n_frequencies)
        The frequencies of the signals.
        
    c : float
        The speed of sound. Defaults to 343.0 m/s.

    Returns
    -------
    mapper : np.array (n_candidate_positions, n_microphones, n_microphones, n_frequencies) 
    """

    mapper = tdoa_mapper(candidate_grid, microphone_positions)[..., np.newaxis]
    # mapper.shape == (n_candidate_positions, n_microphones, n_microphones, 1)
    mapper = np.repeat(mapper, len(freqs), axis=-1)
    # mapper.shape == (n_candidate_positions, n_microphones, n_microphones, n_frequencies)
    mapper[..., :] *= freqs[np.newaxis, np.newaxis, np.newaxis, :]
    # mapper.shape == (n_candidate_positions, n_microphones, n_microphones, n_frequencies)
    # The measured phase is exp(-j*w*tau). We need to multiply by exp(j*w*tau) to align them.
    # So we use positive sign here.
    mapper = np.exp(2j*mapper*np.pi)
    return mapper


def _compute_theoretical_tdoa(candidate_grid, mic_0, mic_1, c=343.0, far_field=False):
    """Compute the theoretical TDOA between a grid of candidate positions and a microphone pair.
    
    For near-field (positional grids): TDOA is computed as the difference between the Time of 
    Flight (TOF) of the candidate position to the first microphone and the TOF of the candidate 
    position to the second microphone.
    
    For far-field (DOA grids): TDOA is computed using the projection of the microphone baseline
    onto the direction vector, assuming planar wavefronts.

    Parameters
    ----------
    candidate_grid : np.array (n_points, n_dimensions)
        The candidate positions (physical positions for near-field, unit direction vectors for far-field).
    mic_0 : np.array (n_dimensions)
        The first microphone.
    mic_1 : np.array (n_dimensions)
        The second microphone.
    c : float
        The speed of sound. Defaults to 343.0 m/s.
    far_field : bool
        If True, use far-field (planar wave) model. If False, use near-field (spherical wave) model.
        Defaults to False.
    
    Returns
    -------
    tdoa : np.array (n_points,)
        The theoretical TDOA between the candidate grid and the microphone pair.
    """

    if far_field:
        # Far-field model: TDOA = (baseline . direction) / c
        # where baseline = mic_1 - mic_0 and direction is the unit vector
        # The sign convention: positive TDOA means signal arrives at mic_0 before mic_1
        # For a source in direction u, the wavefront hits the closer mic first
        baseline = mic_1 - mic_0
        # Project baseline onto direction vectors
        # TDOA = t_0 - t_1 = (baseline . u) / c
        tdoa = np.dot(candidate_grid, baseline) / c
    else:
        # Near-field model: compute exact time-of-flight differences
        tof_0 = np.linalg.norm(candidate_grid - mic_0, axis=-1)/c
        tof_1 = np.linalg.norm(candidate_grid - mic_1, axis=-1)/c
        tdoa = tof_0 - tof_1

    return tdoa


def compute_tdoa_matrix(mic_positions, source_position, speed_of_sound=343.0, fs=None):
    """Compute the time difference of arrival between a source and a set of microphones.
    
    Parameters
    ----------
    mic_positions : np.ndarray (n_mics, 3)
        The positions of the microphones.
    source_position : np.ndarray (3,)
        The position of the source.
    speed_of_sound : float
        The speed of sound in the medium.
    fs : float
        The sampling frequency of the signal.
        If provided, the time difference of arrival will be converted to samples.
    Returns
    -------
    np.ndarray (n_mics, n_mics)
        The time difference of arrival between the source and the microphones.
    """

    # Compute the distance between the source and the microphones
    distances = np.linalg.norm(mic_positions - source_position, axis=1)

    # Compute the time difference of arrival between the source and the microphones
    tdoa = distances/speed_of_sound

    # Compute the TDOA matrix
    tdoa_matrix = np.zeros((len(tdoa), len(tdoa)))
    for i in range(len(tdoa)):
        for j in range(len(tdoa)):
            tdoa_matrix[i, j] = tdoa[j] - tdoa[i]

    if fs is not None:
        tdoa_matrix *= fs
    
    return tdoa_matrix
