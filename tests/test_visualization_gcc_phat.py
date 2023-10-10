import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf

from visualization.cross_correlation import (
    plot_cross_correlation
)
from xsrp.signal_features.cross_correlation import cross_correlation

from xsrp.signal_features.gcc_phat import gcc_phat


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
    

def test_visualize_gcc_phat_vs_cross_correlation():
    os.makedirs("tests/temp", exist_ok=True)

    sr = 16000
    
    # Create a one second signal
    signal_1 = sf.read("tests/fixtures/p225_001.wav")[0]
    # Add a delay of 2ms to the second signal
    delay = int(sr*0.002)
    
    signal_2 = np.concatenate((np.zeros(delay), signal_1[:-delay]))

    signals = np.stack((signal_1, signal_2))

    gcc_matrix, lags = gcc_phat(signals, ifft=True, abs=False, return_lags=True)
    cc_matrix, lags = cross_correlation(signals)

    fig, ax = plt.subplots(figsize=(4, 2))

    axs_gcc = plot_cross_correlation(gcc_matrix, sr=16000,
                                     plot_peaks=False,
                                     lags=lags, n_central_bins=128, axs=ax, label="GCC-PHAT")
    axs_cc = plot_cross_correlation(cc_matrix, sr=16000,
                                    plot_peaks=False,
                                    lags=lags, n_central_bins=128, axs=ax, label="Cross-correlation")

    ax.set_title("GCC-PHAT vs. cross-correlation")
    ax.legend()

         
    fig.tight_layout()
    fig.savefig("tests/temp/plot_gcc_phat_vs_cross_correlation.pdf")


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
