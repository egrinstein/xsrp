import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf

from xsrp.visualization.cross_correlation import (
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


def test_cross_correlation_multiple_signals():
    # Compare the autocorrelation of
    # three different signals: a broadband noise signal,
    # a broadband speech signal,
    # a low-frequency sine wave, and a high-frequency sine wave

    os.makedirs("tests/temp", exist_ok=True)

    # Load a speech signal
    speech_signal, sr = sf.read("tests/fixtures/p225_001.wav")
    speech_signal = speech_signal[:sr]

    # Create a one second signal
    noise_signal = np.random.randn(sr)

    # Create a one second sine wave with a frequency of 100 Hz
    low_freq_signal = np.sin(2*np.pi*100*np.arange(sr)/sr)
    # Create a one second sine wave with a frequency of 5000 Hz
    high_freq_signal = np.sin(2*np.pi*5000*np.arange(sr)/sr)

    fig, axs = plt.subplots(nrows=4, figsize=(5, 10))

    labels = ["Noise", "Speech", "100 Hz sine", "5 kHz sine"]
    colors = ["C0", "C1", "C2", "C3"]

    for i, (signal, label, color) in enumerate(zip(
        [noise_signal, speech_signal, low_freq_signal, high_freq_signal],
        labels,
        colors
    )):
        signal = np.stack((signal, signal))
        cross_correlation_matrix, lags = cross_correlation(signal)
        gcc_phat_matrix, _ = gcc_phat(signal)
        axs[i].plot(lags, cross_correlation_matrix[0, 0], label="Cross-correlation")
        axs[i].plot(lags, gcc_phat_matrix[0, 0], label="GCC-PHAT")
        axs[i].set_ylabel("Value")
        axs[i].set_title(label)
        
        # Set xticks off for all but the bottom plot
        if i != len(labels) - 1:
            axs[i].set_xticks([])
            if i == 0:
                axs[i].legend()
        else:
            axs[i].set_xlabel("Delay (samples)")

    fig.savefig("tests/temp/cross_correlation_multiple_signals.png")
