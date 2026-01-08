import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra

from xsrp.visualization.grids import plot_uniform_cartesian_grid
from xsrp.xsrp import XSrp


def test_compare_averaging():
    os.makedirs("tests/temp", exist_ok=True)

    fs, room_dims, mic_positions, source_position, signals = _simulate()

    dibiase_srp_no_avg = XSrp(fs, "2D", 50,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=True,
        n_average_samples=1)
    dibiase_srp_avg = XSrp(fs, "2D", 50,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=True,
        n_average_samples=5)
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_no_avg.forward(signals)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[0],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_avg.forward(signals)

    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[1],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    axs[0].set_title("No averaging")
    axs[1].set_title("5-sample averaging")

    plt.savefig("tests/temp/srp_compare_averaging.png")


def test_compare_interpolation():
    os.makedirs("tests/temp", exist_ok=True)

    fs, room_dims, mic_positions, source_position, signals = _simulate()

    dibiase_srp_no_avg = XSrp(fs, "2D", 50,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False)
    dibiase_srp_avg = XSrp(fs, "2D", 50,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=True)
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_no_avg.forward(signals)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[0],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_avg.forward(signals)

    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[1],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    axs[0].set_title("No interpolation")
    axs[1].set_title("Parabolic interpolation")

    plt.savefig("tests/temp/srp_compare_interpolation.png")


def test_compare_gcc_and_cc():
    os.makedirs("tests/temp", exist_ok=True)

    fs, room_dims, mic_positions, source_position, signals = _simulate()
    n_grid = 100

    dibiase_srp_cc = XSrp(fs, "2D", n_grid,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False,
        mode="cross_correlation")
    dibiase_srp_gcc = XSrp(fs, "2D", n_grid,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False,
        mode="gcc_phat_time")
    dibiase_srp_gcc_freq = XSrp(fs, "2D", n_grid,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False,
        mode="gcc_phat_freq")
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_cc.forward(signals)

    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[0],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_gcc.forward(signals)

    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[1],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp_gcc_freq.forward(signals)

    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, ax=axs[2],
                                mic_positions=mic_positions,
                                source_positions=source_position[np.newaxis, :])
    
    axs[0].set_title("Cross-correlation")
    axs[1].set_title("GCC-PHAT")
    axs[2].set_title("GCC-PHAT-FREQ")

    plt.savefig("tests/temp/srp_compare_cc_gcc.png")


def _simulate():
    fs = 16000

    # Create a room with sources and microphones
    room_dims = [10, 10]

    # Place four microphones randomly within the room
    mic_positions = np.random.uniform(low=0, high=10, size=(4, 2))    

    # Place the source randmly within the room
    source_position = np.random.uniform(low=0, high=10, size=(2,))
    source_signal = np.random.normal(size=fs)

    room = pra.ShoeBox(room_dims, fs=fs, max_order=0)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(
        pra.MicrophoneArray(
            mic_positions.T, room.fs
        )
    )
    room.simulate()
    signals = room.mic_array.signals

    return fs, room_dims, mic_positions, source_position, signals