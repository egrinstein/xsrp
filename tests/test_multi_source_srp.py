import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from xsrp.multi_source_srp import notch_filter, de_emphasize_peak
from xsrp.signal_features.cross_correlation import cross_correlation


def test_notch_filter():
    r = np.linspace(-20, 20, 200)

    mu = 10
    b = 2
    p = 2

    window = notch_filter(r, mu, b, p)
    window /= np.sum(window)

    plt.plot(r, window)
    plt.title(f"Notch filter centered at {mu}")
    plt.xlabel("index")
    plt.ylabel("value")

    plt.savefig("tests/temp/notch_filter.png")


def test_de_emphasize_peak():
    os.makedirs("tests/temp", exist_ok=True)



    # Create a room with sources and microphones
    room_dims = [10, 10, 10]

    # Place the source near the middle of the room
    source_position = [5, 5, 5]
    source_signal, fs = sf.read("tests/fixtures/p225_001.wav")

    # Place the microphones in the corners of the room
    mic_positions = np.array([
        [1, 1, 1],
        [1, 9, 1],
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

    cc, lags = cross_correlation(signals,)
    cc = cc[0, 1]
    # De-emphasize the peak at the centre of the cross-correlation matrix
    cc_new, window = de_emphasize_peak(cc, lags, 0, b=10, p=2,
                                       return_notch_filter=True)

    fig, axs = plt.subplots(3, 1, sharex=True)

    axs[0].set_title(f"Original cross-correlation function")
    axs[0].plot(lags, cc)
    axs[0].set_ylabel("Cross-correlation")

    axs[2].set_title(f"De-emphasized cross-correlation function")
    axs[1].plot(lags, cc_new)
    axs[1].set_ylabel("Cross-correlation")
    
    axs[0].set_title(f"Notch filter")
    axs[2].set_ylabel("Cross-correlation")
    axs[2].set_xlabel("Delay")

    axs[2].plot(lags, window)

    plt.tight_layout()
    plt.savefig("tests/temp/de_emphasize_peak.png")
