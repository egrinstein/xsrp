
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlation_lags
from scipy.signal import find_peaks


def plot_cross_correlation(cross_correlation_matrix, plot_peaks=False,
                           n_central_bins=None, lags=None, output_path="",
                           sr=None, axs=None, label=None):
    """Plot the cross-correlation function between two or more signals.
    
    Parameters
    ----------
    cross_correlation_matrix : np.ndarray (n_signals, n_signals, n_samples)
    """

    n_signals, _, n_samples = cross_correlation_matrix.shape
    
    if lags is None:
        lags = correlation_lags(n_samples, n_samples)
    else:
        lags = lags

    if n_central_bins is None:
        n_central_bins = n_samples

    x_central = lags[len(lags)//2-n_central_bins//2:len(lags)//2+n_central_bins//2]
    if sr is not None:
        x_central = x_central/sr*1000 # Convert to ms

    n_pairs = n_signals*(n_signals - 1)//2
    
    show_plot = False
    if axs is None:
        fig, axs = plt.subplots(nrows=n_pairs, figsize=(5, 2*n_pairs))
        show_plot = True
    if n_pairs == 1:
        axs = np.expand_dims(axs, axis=0)

    n_pair = 0
    for i in range(n_signals):
        for j in range(i + 1, n_signals):

            corr = np.abs(cross_correlation_matrix[i, j])
            corr = corr[len(corr)//2-n_central_bins//2:len(corr)//2+n_central_bins//2]
            axs[n_pair].plot(x_central, corr, label=label)
            if plot_peaks:
                peaks, _ = find_peaks(corr)
                axs[n_pair].plot(x_central[peaks], corr[peaks], "x", label="Peaks")
                axs[n_pair].legend()

            axs[n_pair].set_title("Cross-correlation between signals {} and {}".format(i, j))
            axs[n_pair].set_xlabel("Time (ms)")
            axs[n_pair].set_ylabel("Value")
            
            n_pair += 1
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    elif show_plot:
        plt.show()

    return axs
