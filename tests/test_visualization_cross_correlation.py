import numpy as np
import os
import pyroomacoustics as pra

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
    

