import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from visualization.cross_correlation import (
    plot_cross_correlation,
)

from xsrp.signal_features.cross_correlation import cross_correlation


def test_visualize_cross_correlation_matrix():
    os.makedirs("tests/temp", exist_ok=True)

    fs = 16000

    # Create a room with sources and microphones
    room_dims = [10, 10, 10]

    # Place the source near the middle of the room
    source_position = [4, 5, 5]
    source_signal = np.random.normal(size=fs)

    # Place the microphones in the corners of the room
    mic_positions = np.array([
        [1, 1, 1],
        [1, 9, 1],
        [9, 1, 1],
        [9, 9, 1]
    ])

    room = pra.ShoeBox(room_dims, fs=fs, max_order=0)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(
        pra.MicrophoneArray(
            mic_positions.T, room.fs
        )
    )
    room.simulate()
    signals = room.mic_array.signals

    cross_correlation_matrix, lags = cross_correlation(signals)

    plot_cross_correlation(cross_correlation_matrix, sr=16000,
                        plot_peaks=False, output_path="tests/temp/plot_cross_correlation_simulation.png",
                        lags=lags, n_central_bins=256)




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

    labels = ["Noise", "Speech", "Low frequency", "High frequency"]
    colors = ["C0", "C1", "C2", "C3"]

    for i, (signal, label, color) in enumerate(zip(
        [noise_signal, speech_signal, low_freq_signal, high_freq_signal],
        labels,
        colors
    )):
        cross_correlation_matrix, lags = cross_correlation(np.stack((signal, signal)))
        axs[i].plot(lags, cross_correlation_matrix[0, 0], label=label, color=color)
        axs[i].set_ylabel("Cross-correlation")
        axs[i].set_title(label)
        
        # Set xticks off for all but the bottom plot
        if i != len(labels) - 1:
            axs[i].set_xticks([])
        else:
            axs[i].set_xlabel("Delay (samples)")

    fig.savefig("tests/temp/cross_correlation_multiple_signals.png")
