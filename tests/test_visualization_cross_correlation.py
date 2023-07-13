import numpy as np
import os

from visualization.cross_correlation import (
    plot_cross_correlation,
)

from signal_features.cross_correlation import cross_correlation


def test_visualize_cross_correlation():
    os.makedirs("tests/temp", exist_ok=True)

    sr = 16000
    
    # Create a one second signal
    signal_1 = np.random.randn(sr)
    # Add a delay of 2ms to the second signal
    delay = int(sr*0.002)
    
    signal_2 = np.concatenate((np.zeros(delay), signal_1[:-delay]))

    signals = np.stack((signal_1, signal_2))

    cross_correlation_matrix, lags = cross_correlation(signals)

    plot_cross_correlation(cross_correlation_matrix, sr=16000,
                           plot_peaks=False, output_path="tests/temp/plot_cross_correlation.png",
                           lags=lags, n_central_bins=256)
    
