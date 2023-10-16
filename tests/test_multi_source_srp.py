import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from xsrp.spatial_mappers import compute_tdoa_matrix
from xsrp.grid_search import notch_filter, de_emphasize_peak
from xsrp.signal_features.gcc_phat import gcc_phat
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

    signals, config = _simulate_multi_source()

    # Compute the TDOA matrix for first source
    mic_positions = np.array(config["mic_positions"])
    source_positions = np.array(config["source_positions"])
    tdoa_0 = compute_tdoa_matrix(mic_positions,
                                 source_positions[0],
                                 fs=16000)[0, 1]
    tdoa_1 = compute_tdoa_matrix(mic_positions,
                                 source_positions[1],
                                 fs=16000)[0, 1]

    cc, lags = gcc_phat(signals)
    cc = cc[0, 1]

    # Get central bins
    n_central_bins = 100
    cc = cc[len(cc)//2-n_central_bins//2:len(cc)//2+n_central_bins//2]
    lags = lags[len(lags)//2-n_central_bins//2:len(lags)//2+n_central_bins//2]

    # De-emphasize the peak at the centre of the cross-correlation matrix
    cc_new, window = de_emphasize_peak(cc, lags, tdoa_0, b=2, p=2,
                                       return_notch_filter=True)

    fig, axs = plt.subplots(3, 1, sharex=True)

    cc_argmax = np.argmax(cc)
    axs[0].set_title(f"Original cross-correlation function")
    axs[0].plot(lags, cc, label=f"Peak at {lags[cc_argmax]}")
    axs[0].axvline(tdoa_0, color="red", linestyle="--", label="TDOA 1")
    axs[0].axvline(tdoa_1, color="red", linestyle="--", label="TDOA 2")
    axs[0].set_ylabel("Cross-correlation")
    axs[0].legend()

    cc_new_argmax = np.argmax(cc_new)
    axs[1].set_title(f"De-emphasized cross-correlation function")
    axs[1].plot(lags, cc_new, label=f"Peak at {lags[cc_new_argmax]}")
    axs[1].axvline(tdoa_0, color="red", linestyle="--", label="TDOA 1")
    axs[1].axvline(tdoa_1, color="red", linestyle="--", label="TDOA 2")
    axs[1].set_ylabel("Cross-correlation")
    axs[1].legend()
    
    axs[2].plot(lags, window)
    axs[2].axvline(tdoa_0, color="red", linestyle="--", label="TDOA 1")
    axs[2].set_title(f"Notch filter")
    axs[2].set_ylabel("Weight")
    axs[2].set_xlabel("Delay")

    plt.tight_layout()
    plt.savefig("tests/temp/de_emphasize_peak.png")


def _simulate_multi_source():
    fs = 16000

    config = {
        "room_dims": [8, 4, 4],
        "rt60": .15,
        "source_positions": [
            [1, 1, 2],
            [6, 1, 2],
        ],
        "source_signals": [
            "tests/fixtures/46.wav",
            "tests/fixtures/12546.wav",
        ],
        "mic_positions": [
            [4.1, 2, 2],
            [3.9, 2, 2],
        ],
    }

    # Create a room with sources and microphones
    e_absorption, max_order = pra.inverse_sabine(config["rt60"],
                                                 config["room_dims"])

    room = pra.ShoeBox(config["room_dims"], fs=fs,
                       materials=pra.Material(e_absorption),
                       max_order=max_order)
    
    # Place sources
    for source_position, source_signal in zip(config["source_positions"],
                                              config["source_signals"]):
        source_signal, _ = sf.read(source_signal)
        room.add_source(source_position, signal=source_signal)

    room.add_microphone_array(  
        pra.MicrophoneArray(
            np.array(config["mic_positions"]).T, fs=fs
        )
    )
    room.simulate()

    return room.mic_array.signals, config
