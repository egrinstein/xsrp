import numpy as np

from scipy.signal import correlation_lags

from .grids import Grid


def integer_sample_projector(grid: Grid,
                             spatial_mapper: np.array,
                             cross_correlation_matrix: np.array,
                             sum_pairs: bool = True):
    """
    
    Creates a Steered Response Power (SRP) likelihood map by associating
    a cross-correlation value for each grid cell with the
    corresponding spatial location.

    Args:
        grid (Grid): Grid object of n_cells
        spatial_mapper (np.array): Array of shape (n_cells, n_mics, n_mics)
        cross_correlation_matrix (np.array): Array of cross-correlation values
            of shape (n_mics, n_mics, n_samples)
        sum_pairs (bool, optional): Whether to sum the cross-correlation values
            of each microphone pair. Defaults to True.

    Returns:
        srp_map (np.array): Array of shape (n_cells, n_mics, n_mics) if sum_pairs
            is False, else array of shape (n_cells,)
    """

    # Get the number of samples
    n_samples = cross_correlation_matrix.shape[-1]

    n_mics = spatial_mapper.shape[1]

    lags = correlation_lags(n_samples/2, n_samples/2)
    lags_to_idx = dict(
        zip(lags, range(len(lags)))
    ) 

    # if sum_pairs:
    #     srp_map = np.zeros(len(grid))
    # else:
    srp_map = np.zeros((len(grid), n_mics, n_mics))

    # Loop over the cells
    for n_cell in range(len(grid)):
        # Loop over the microphone pairs
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                # Get the cross-correlation integer delay value for the current cell and mic pair
                cross_correlation_lag = spatial_mapper[n_cell, i, j]
                cross_correlation_idx = lags_to_idx[cross_correlation_lag]
                # Get the cross-correlation value for the current cell and mic pair
                cross_correlation_value = cross_correlation_matrix[i, j, cross_correlation_idx]

                if sum_pairs:
                    srp_map[n_cell] += cross_correlation_value
                else:
                    srp_map[n_cell, i, j] = cross_correlation_value

    if sum_pairs:
        srp_map = srp_map.sum(axis=1).sum(axis=1)

    return srp_map
