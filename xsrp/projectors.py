import numpy as np

from scipy.signal import correlation_lags
from scipy.interpolate import interp1d

from .grids import Grid


def average_sample_projector(grid: Grid,
                             spatial_mapper: np.array,
                             cross_correlation_matrix: np.array,
                             sum_pairs: bool = True,
                             n_average_samples: int = 1):
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
        n_average_samples (int, optional): Number of samples to average over.
            Defaults to 1.

    Returns:
        srp_map (np.array): Array of shape (n_cells, n_mics, n_mics) if sum_pairs
            is False, else array of shape (n_cells,)
    """

    # Get the number of samples
    n_samples = cross_correlation_matrix.shape[-1]

    n_mics = spatial_mapper.shape[1]

    lags = correlation_lags(n_samples/2, n_samples/2)
    lag_idxs = range(len(lags))
    lag_to_idx = interp1d(lags, lag_idxs, kind="linear")

    srp_map = np.zeros((len(grid), n_mics, n_mics))

    # Loop over the cells
    for n_cell in range(len(grid)):
        # Loop over the microphone pairs
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                cross_correlation_ij = cross_correlation_matrix[i, j]
                # Get the cross-correlation integer delay value for the current cell and mic pair
                cross_correlation_lag = spatial_mapper[n_cell, i, j]
                cross_correlation_idx = lag_to_idx(cross_correlation_lag)

                n_left_neighbours = int(np.floor(n_average_samples/2))
                n_right_neighbours = int(np.ceil(n_average_samples/2))

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


def frequency_projector(grid: Grid, spatial_mapper: np.array,
                        cross_correlation_matrix: np.array,
                        sum_pairs: bool = True):
    """
    Creates a Steered Response Power (SRP) likelihood map by steering the frequency domain
    cross-correlation between microphone pairs to the corresponding grid cell and frequency bin.    

    Args:
        grid (Grid): Grid object of n_cells
        spatial_mapper (np.array): Array of shape (n_cells, n_mics, n_mics)
        cross_correlation_matrix (np.array): Array of cross-correlation values
            of shape (n_mics, n_mics, n_frequencies)
        sum_pairs (bool, optional): Whether to sum the cross-correlation values
            of each microphone pair and frequency. Defaults to True.

    Returns:
        srp_map (np.array): Array of shape (n_cells, n_mics, n_mics) if sum_pairs
            is False, else array of shape (n_cells,)
    """

    n_mics = spatial_mapper.shape[1]
    n_frequencies = cross_correlation_matrix.shape[-1]

    srp_map = cross_correlation_matrix[np.newaxis] * spatial_mapper

    if sum_pairs:
        # Sum the cross-correlation values of each microphone pair and frequency
        #srp_map = srp_map.sum(axis=1).sum(axis=1)[:, 2]
        srp_map = srp_map.sum(axis=1).sum(axis=1).sum(axis=1)
    return np.abs(srp_map)

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
