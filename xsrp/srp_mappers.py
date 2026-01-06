import numpy as np

from scipy.signal import correlation_lags
from scipy.interpolate import interp1d

from .grids import Grid
from .spatial_mappers import (
    frequency_delay_mapper,
    fractional_sample_mapper,
    integer_sample_mapper
)


def temporal_projector(mic_positions: np.array,
                       candidate_grid: Grid,
                       cross_correlation_matrix: np.array,
                       fs: float,
                       sum_pairs: bool = True,
                       n_average_samples: int = 1,
                       interpolate: bool = False):
    """
    Creates a Steered Response Power (SRP) likelihood map by associating
    a cross-correlation value for each grid cell with the
    corresponding spatial location.

    Args:
        grid (Grid): Grid object of n_cells
        spatial_mapper (np.array): Array of shape (n_cells, n_mics, n_mics)
        cross_correlation_matrix (np.array): Array of cross-correlation values
            of shape (n_mics, n_mics, n_samples)
        fs (float, optional): The sampling rate of the signals.
        sum_pairs (bool, optional): Whether to sum the cross-correlation values
            of each microphone pair. Defaults to True.
        n_average_samples (int, optional): Number of samples to average over.
            Defaults to 1.
        interpolate (bool, optional): Whether to use interpolation to estimate
            the cross-correlation value of each candidate position. Defaults to False.


    Returns:
        srp_map (np.array): Array of shape (n_cells, n_mics, n_mics) if sum_pairs
            is False, else array of shape (n_cells,)
    """

    if interpolate:
        spatial_mapper = fractional_sample_mapper(candidate_grid, mic_positions, fs)
    else:
        spatial_mapper = integer_sample_mapper(candidate_grid, mic_positions, fs)


    # Get the number of samples
    n_samples = cross_correlation_matrix.shape[-1]

    n_mics = spatial_mapper.shape[1]

    lags = correlation_lags(n_samples/2, n_samples/2)
    lag_idxs = range(len(lags))
    lag_to_idx = interp1d(lags, lag_idxs, kind="linear")

    srp_map = np.zeros_like(spatial_mapper, dtype=float)

    # Loop over the cells
    for n_cell in range(len(spatial_mapper)):
        # Loop over the microphone pairs
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                cross_correlation_ij = cross_correlation_matrix[i, j]
                # Get the cross-correlation integer delay value for the current cell and mic pair
                cross_correlation_lag = spatial_mapper[n_cell, i, j]
                cross_correlation_idx = lag_to_idx(cross_correlation_lag)

                n_left_neighbours = int(np.floor(n_average_samples/2))
                n_right_neighbours = int(np.ceil(n_average_samples/2))

                if n_average_samples == 0:
                    # No averaging: use exact sample value (nearest neighbor)
                    cross_correlation_idx_int = int(np.round(cross_correlation_idx))
                    # Clamp to valid range
                    cross_correlation_idx_int = max(0, min(len(cross_correlation_ij) - 1, cross_correlation_idx_int))
                    cross_correlation_value = cross_correlation_ij[cross_correlation_idx_int]
                elif n_left_neighbours + n_right_neighbours == 1:
                    # If only one neighbour is needed, only use interpolation
                    cross_correlation_value = _parabolic_interpolation(cross_correlation_idx, cross_correlation_ij)
                else:
                    # Get the cross-correlation values of the neighbouring samples
                    cross_correlation_values = cross_correlation_ij[
                        int(cross_correlation_idx - n_left_neighbours):
                        int(cross_correlation_idx + n_right_neighbours)
                    ]

                    # Average the cross-correlation values
                    cross_correlation_value = cross_correlation_values.mean()

                if sum_pairs:
                    srp_map[n_cell] += cross_correlation_value
                else:
                    srp_map[n_cell, i, j] = cross_correlation_value

    if sum_pairs:
        # Sum the cross-correlation values of each microphone pair
        srp_map = srp_map.sum(axis=1).sum(axis=1)

    return srp_map


def frequency_projector(mic_positions: np.array,
                        candidate_grid: Grid,
                        cross_correlation_matrix: np.array,
                        fs: float,
                        sum_pairs: bool = True,
                        freq_cutoff: int = None,
                        frequency_weights: np.array = None,
                        return_per_freq: bool = False):
    """
    Creates a Steered Response Power (SRP) likelihood map by steering the frequency domain
    cross-correlation between microphone pairs to the corresponding grid cell and frequency bin.    

    Args:

        mic_positions (np.array): Array of shape (n_mics, 3 or 2)
        candidate_grid (Grid): Grid object of n_cells
        cross_correlation_matrix (np.array): Array of cross-correlation values
            of shape (n_mics, n_mics, n_frequencies)
        fs (float, optional): The sampling rate of the signals.
        sum_pairs (bool, optional): Whether to sum the cross-correlation values
            of each microphone pair and frequency. Defaults to True.
        freq_cutoff (int, optional): The frequency bin number from which to cutoff
            the cross-correlation values. Defaults to None.
        frequency_weights (np.array, optional): Array of shape (n_frequencies,) to weight
            each frequency bin before summation. Defaults to None (uniform weighting).
        return_per_freq (bool, optional): If True, also return per-frequency SRP maps.
            Defaults to False.

    Returns:
        srp_map (np.array): Array of shape (n_cells, n_mics, n_mics) if sum_pairs
            is False, else array of shape (n_cells,)
        per_freq_srp_map (np.array, optional): Array of shape (n_cells, n_frequencies)
            if return_per_freq is True. Only returned if return_per_freq is True.
    """

    n_freqs = cross_correlation_matrix.shape[-1]
    freqs = np.linspace(0, fs/2, n_freqs)
    spatial_mapper = frequency_delay_mapper(candidate_grid, mic_positions, freqs)

    # Multiply cross-spectrum by steering vector to align phases
    srp_map = cross_correlation_matrix[np.newaxis] * spatial_mapper

    if freq_cutoff is not None:
        srp_map = srp_map[..., :freq_cutoff]
        n_freqs = freq_cutoff

    if sum_pairs:
        # Sum the cross-correlation values of each microphone pair, but keep frequency dimension
        # srp_map shape: (n_cells, n_mics, n_mics, n_freqs)
        
        # Calculate the sum including diagonal
        per_freq_srp_map_all = srp_map.sum(axis=1).sum(axis=1)  # Sum mic pairs, keep freq dim
        
        # Calculate the trace (sum of diagonal elements: auto-correlations)
        # diagonal elements are at indices [:, i, i, :]
        trace = np.trace(srp_map, axis1=1, axis2=2)
        
        # Subtract trace to keep only cross-correlations
        per_freq_srp_map = per_freq_srp_map_all - trace
        
        per_freq_srp_map = np.real(per_freq_srp_map)
        
        # Apply frequency weights if provided
        if frequency_weights is not None:
            if len(frequency_weights) != n_freqs:
                raise ValueError(
                    f"frequency_weights length ({len(frequency_weights)}) must match "
                    f"number of frequency bins ({n_freqs})"
                )
            # Apply weights: (n_cells, n_freqs) * (n_freqs,) -> (n_cells, n_freqs)
            per_freq_srp_map = per_freq_srp_map * frequency_weights[np.newaxis, :]
        
        # Sum across frequencies to get final SRP map
        srp_map = per_freq_srp_map.sum(axis=1)
        
        if return_per_freq:
            return srp_map, per_freq_srp_map
    else:
        srp_map = np.real(srp_map)
    
    return srp_map


def _parabolic_interpolation(x_value, y):
    """Compute the parabolic interpolation of the given x_value by
    finding three points that are closest
    to the given x_value and using these points to compute the parabolic
    interpolation.

    Parameters
    ----------
    x_value : float
        The value to interpolate.
    y : np.array
        The y values.

    Returns
    -------
    interpolated_y_value : float
        The interpolated value.
    """


    # Find the two neighbours closest to the x_value
    x_left = np.floor(x_value).astype(int)
    x_right = np.ceil(x_value).astype(int)

    if x_left == x_right:
        # If the x_value is an integer, return the corresponding y value
        # as interpolation is not needed
        return y[x_left]

    # Find a third point that is closest to the x_value
    if abs(x_value - x_left) < abs(x_value - x_right):
        x_third = x_left - 1
    else:
        x_third = x_right + 1

    # Get the y values of the three points
    y_left = y[x_left]
    y_right = y[x_right]
    y_third = y[x_third]

    # Compute the parabolic interpolation
    a, b, c = _get_parabola_coeffs(
        np.array([x_left, x_right, x_third]),
        np.array([y_left, y_right, y_third])
    )

    interpolated_y_value = a*x_value**2 + b*x_value + c

    return interpolated_y_value


def _get_parabola_coeffs(x, y):
    """Estimate the coefficients of a parabola ax^2 + bx + c
    passing through the points (x[0], y[0]), (x[1], y[1]), (x[2], y[2])

    Credits to https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    """

    if x.shape[0] != 3:
        raise ValueError(
            "This method only works for an input containing 3 values (peak, left and right neighbours")

    denom = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
    a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / denom
    b = (x[2]*x[2] * (y[0] - y[1]) + x[1]*x[1] * (y[2] - y[0]) + x[0]*x[0] * (y[1] - y[2])) / denom
    c = (x[1] * x[2] * (x[1] - x[2]) * y[0] + x[2] * x[0] * (x[2] - x[0]) * y[1] + x[0] * x[1] * (x[0] - x[1]) * y[2]) / denom

    return a, b, c
