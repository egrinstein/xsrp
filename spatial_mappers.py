import numpy as np


def temporal_spatial_mapper(grid, microphone_positions):
    """Create a temporal spatial mapper for the given grid.

    A temporal spatial mapper associates a delay value
    with each grid cell. The delay value is the theoretical Time Difference
    of Arrival (TDOA) between the cell and the microphone pairs.

    Parameters
    ----------
    grid : Grid
        The grid of candidate positions to create the temporal-spatial mapper for.
        The grid is of shape 2D (n_points, 2/3)
    
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.
        
    Returns
    -------
    mapper : np.array 
        The temporal-spatial mapper.
    """

    if grid.ndim != 2:
        raise ValueError("The grid must be 2D.")

    if grid.shape[-1] != microphone_positions.shape[-1]:
        raise ValueError("The grid and microphone positions must have the same number of dimensions.")

    n_candidate_positions, n_dims = grid.shape[0]

    n_microphones = microphone_positions.shape[0]

    mapper = np.zeros((n_candidate_positions, n_microphones, n_microphones))

    for i in range(n_microphones):
        for j in range(n_microphones):
            mapper[:, i, j] = _compute_theoretical_tdoa(grid, microphone_positions[i], microphone_positions[j])

    return mapper


def _compute_theoretical_tdoa(candidate_position, mic_0, mic_1, c=343.0):
    """Compute the theoretical TDOA between a candidate position and a microphone pair.
    The TDOA is computed as the difference between the Time of Flight (TOF) of the candidate position
    to the first microphone and the TOF of the candidate position to the second microphone.
    

    Parameters
    ----------
    candidate_position : np.array (n_dimensions)
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
        The theoretical TDOA between the candidate position and the microphone pair.
    """

    tof_0 = np.linalg.norm(candidate_position - mic_0)/c
    tof_1 = np.linalg.norm(candidate_position - mic_1)/c

    return tof_1 - tof_0
