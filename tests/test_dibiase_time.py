import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra

from visualization.grids import plot_uniform_cartesian_grid
from xsrp.conventional_srp import ConventionalSrp


def test_dibiase_time():
    os.makedirs("tests/temp", exist_ok=True)

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

    dibiase_srp = ConventionalSrp(fs, "2D", 50,
        mic_positions=mic_positions, room_dims=room_dims, interpolation=False)
    
    estimated_positions, srp_map, candidate_grid = dibiase_srp.forward(signals)

    # Plot the srp map
    plot_uniform_cartesian_grid(candidate_grid, room_dims,
                                srp_map=srp_map, output_path="tests/temp/srp_map_dibiase_time.png",
                                mic_positions=mic_positions, source_positions=source_position[np.newaxis, :])