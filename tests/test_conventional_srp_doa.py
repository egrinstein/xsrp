import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra

from visualization.grids import plot_uniform_cartesian_grid
from xsrp.conventional_srp import ConventionalSrp


def test_doa():
    os.makedirs("tests/temp", exist_ok=True)

    fs, room_dims, mic_positions_norm, mic_positions, source_position_norm, source_position, signals = _simulate()

    dibiase_srp = ConventionalSrp(fs, "doa_1D", 360,
        mic_positions=mic_positions_norm, interpolation=False,
        n_average_samples=1)
    dibiase_srp_cart = ConventionalSrp(fs, "2D", 100,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False,
        n_average_samples=1)
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp.forward(signals)

    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, [[-1, 1], [-1, 1]],
                                srp_map=srp_map, ax=axs[0],
                                source_positions=source_position_norm[np.newaxis, :])
    axs[0].set_title("Circular grid")

    estimated_positions, srp_map, candidate_grid = dibiase_srp_cart.forward(signals)

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, [[-1, 1], [-1, 1]],
                                srp_map=srp_map, ax=axs[1],
                                source_positions=source_position[np.newaxis, :])
    axs[1].set_title("Cartesian grid")


    plt.savefig("tests/temp/srp_doa.png")




def _simulate():
    fs = 16000

    # Create a room with sources and microphones
    room_dims = [10, 10]

    # Place a microphone array in the room  
    mic_positions_norm = pra.circular_2D_array([0, 0], 8, 0, 0.05).T
    mic_positions = mic_positions_norm + np.array([5, 5])
    # Place the source 1 meter away from the microphone array at a random angle
    
    angle = np.random.uniform(low=0, high=2*np.pi)
    source_position_norm = np.array([np.cos(angle), np.sin(angle)])
    source_position = source_position_norm + np.array([5, 5])
    source_signal = np.random.normal(size=fs)

    room = pra.ShoeBox(room_dims, fs=fs, max_order=0)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(mic_positions.T)
        
    room.simulate()
    signals = room.mic_array.signals

    return fs, room_dims, mic_positions_norm, mic_positions, source_position_norm, source_position, signals