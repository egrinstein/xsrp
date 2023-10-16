import numpy as np

from scipy.signal import correlate, correlation_lags


def cross_correlation(signals, n_central_bins=None, abs=True, return_lags=True, normalize=True):
    """Compute the cross-correlation between two or more signals.
    
    Parameters
    ----------
    signals : np.ndarray (n_signals, n_samples)
        The signals to correlate.
    n_central_bins : int
        The number of bins to keep in the central part of the cross-correlation.
    abs : bool
        Whether to take the absolute value of the cross-correlation.
    return_lags : bool
        Whether to return the lags of the cross-correlation.
    """

    if n_central_bins is None:
        n_central_bins = signals.shape[1]//2
        
    n_signals, n_samples = signals.shape
    if n_signals < 2:
        raise ValueError("At least two signals must be provided.")
    
    x_corr = correlation_lags(n_samples, n_samples)
    x_central = x_corr[len(x_corr)//2-n_central_bins//2:len(x_corr)//2+n_central_bins//2]
    
    cross_correlation_matrix = []

    for i in range(n_signals):
        cross_correlation_matrix.append([])
        for j in range(n_signals):
            corr = correlate(signals[i], signals[j])
            corr = corr[len(corr)//2-n_central_bins//2:len(corr)//2+n_central_bins//2]

            if abs:
                corr = np.abs(corr)
            
            cross_correlation_matrix[i].append(corr)

    cross_correlation_matrix = np.array(cross_correlation_matrix)

    if normalize:
        cross_correlation_matrix = cross_correlation_matrix/np.max(np.abs(cross_correlation_matrix))

    if return_lags:
        return cross_correlation_matrix, x_central
    else:
        return cross_correlation_matrix
