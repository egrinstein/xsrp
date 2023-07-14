import numpy as np

from scipy.signal import correlation_lags


def gcc_phat(signals, abs=True, return_lags=True, ifft=True, n_dft_bins=None):
    """Compute the generalized cross-correlation with phase transform (GCC-PHAT) between two or more signals.
    
    Parameters
    ----------
    signals : np.ndarray (n_signals, n_samples)
        The signals to correlate.
    abs : bool
        Whether to take the absolute value of the cross-correlation. Only used if ifft is True.
    return_lags : bool
        Whether to return the lags of the cross-correlation. Only used if ifft is True.
    ifft : bool
        Whether to use the inverse Fourier transform to compute the cross-correlation in the time domain,
        instead of returning the cross-correlation in the frequency domain.
    n_dft_bins : int
        The number of DFT bins to use. If None, the number of DFT bins is set to n_samples//2 + 1.
    """

    n_bins = signals.shape[1]//2 + 1
    
    if n_dft_bins is None:
        n_dft_bins = n_bins

    n_signals, n_samples = signals.shape
    if n_signals < 2:
        raise ValueError("At least two signals must be provided.")
    
    x_corr = correlation_lags(n_samples, n_samples)
    x_central = x_corr[len(x_corr)//2-n_bins:len(x_corr)//2+n_bins]

    signals_dft = np.fft.rfft(signals, n=n_dft_bins)

    gcc_phat_matrix = []

    for i in range(n_signals):
        gcc_phat_matrix.append([])
        for j in range(n_signals):
            gcc_ij = signals_dft[i]*np.conj(signals_dft[j])
            gcc_phat_ij = gcc_ij/np.abs(gcc_ij)

            if ifft:
                gcc_phat_ij = np.fft.irfft(gcc_phat_ij)
                if abs:
                    gcc_phat_ij = np.abs(gcc_phat_ij)
            
                gcc_phat_ij = np.concatenate((gcc_phat_ij[len(gcc_phat_ij)//2:],
                                                gcc_phat_ij[:len(gcc_phat_ij)//2]))

            gcc_phat_matrix[-1].append(gcc_phat_ij)

    gcc_phat_matrix = np.array(gcc_phat_matrix)

    if ifft and return_lags:
        return gcc_phat_matrix, x_central
    else:
        return gcc_phat_matrix
