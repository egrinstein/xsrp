import matplotlib.pyplot as plt
import numpy as np
import os

from visualization.cross_correlation import (
    plot_cross_correlation
)

from signal_features.gcc_phat import gcc_phat


def test_visualize_gcc_phat_time():
    os.makedirs("tests/temp", exist_ok=True)

    sr = 16000
    
    # Create a one second signal
    signal_1 = np.random.randn(sr)
    # Add a delay of 2ms to the second signal
    delay = int(sr*0.002)
    
    signal_2 = np.concatenate((np.zeros(delay), signal_1[:-delay]))

    signals = np.stack((signal_1, signal_2))

    cross_correlation_matrix, lags = gcc_phat(signals, ifft=True, abs=False, return_lags=True)

    plot_cross_correlation(cross_correlation_matrix, sr=16000,
                           plot_peaks=False, output_path="tests/temp/plot_gcc_phat_time.png",
                           lags=lags, n_central_bins=128)
    

def test_visualize_gcc_phat_freq():
    os.makedirs("tests/temp", exist_ok=True)

    sr = 16000
    
    # Create a one second signal
    signal_1 = np.random.randn(sr)
    # Add a delay of 2ms to the second signal
    delay = int(sr*0.002)
    
    signal_2 = np.concatenate((np.zeros(delay), signal_1[:-delay]))

    signals = np.stack((signal_1, signal_2))

    cross_correlation_matrix = gcc_phat(signals, ifft=False)

    plt.plot(np.angle(cross_correlation_matrix[0, 1]))
    plt.savefig("tests/temp/plot_gcc_phat_freq.png")
