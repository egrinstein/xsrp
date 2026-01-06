import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from visualization.grids import (
    plot_azimuth_elevation_grid, plot_uniform_cartesian_grid,
    plot_azimuth_elevation_heatmap

)
from xsrp.conventional_srp import ConventionalSrp


# def test_doa_2d():
#     os.makedirs("tests/temp", exist_ok=True)

#     fs, room_dims, mic_positions_norm, mic_positions, source_position_norm, source_position, signals = _simulate_2d()

#     srp_func = ConventionalSrp(fs, "doa_1D", 180,
#         mic_positions=mic_positions_norm, interpolation=False,
#         n_average_samples=1)
#     srp_func_cart = ConventionalSrp(fs, "2D", 100,
#         mic_positions=mic_positions, room_dims=room_dims, interpolation=False,
#         n_average_samples=1)
    
#     estimated_positions, srp_map, candidate_grid = srp_func.forward(signals)

#     fig, axs = plt.subplots(ncols=2, figsize=(10, 5))

#     # Plot the srp map
#     plot_uniform_cartesian_grid(candidate_grid, [[-1, 1], [-1, 1]],
#                                 srp_map=srp_map, ax=axs[0],
#                                 source_positions=source_position_norm[np.newaxis, :])
#     axs[0].set_title("Circular grid")

#     estimated_positions, srp_map, candidate_grid = srp_func_cart.forward(signals)

#     # Plot the srp map
#     plot_uniform_cartesian_grid(candidate_grid, [[-1, 1], [-1, 1]],
#                                 srp_map=srp_map, ax=axs[1],
#                                 source_positions=source_position[np.newaxis, :])
#     axs[1].set_title("Cartesian grid")


#     plt.savefig("tests/temp/srp_doa.png")


# def test_doa_3d():
#     os.makedirs("tests/temp", exist_ok=True)

#     # Create 3 scenarios: low-reverb without interference, high-reverb without interference and low-reverb with interference

#     fig, axs = plt.subplots(nrows=3, figsize=(5, 8))

#     # 1. Low-reverb without interference
#     (
#         fs, _,
#         mic_positions_norm, _,
#         source_position_norm, _,
#         interferer_position_norm, _,
#         signals,
#     ) = base_scenario = _simulate_3d(interferer=True)
#     base_scenario = base_scenario[1:-1]
    
#     signals_base = _simulate_3d(interferer=False, scenario=base_scenario)[-1]

#     srp_func = ConventionalSrp(fs, "doa_2D", 200,
#         mic_positions=mic_positions_norm, interpolation=False,
#         n_average_samples=2)

#     estimated_positions, srp_map_base, candidate_grid = srp_func.forward(signals_base)

#     RT60 = 0.2
#     axs[0].set_title(f"(a) RT60 of {RT60}s")
#     plot_azimuth_elevation_grid(
#         candidate_grid, srp_map=srp_map_base,
#         source_positions=source_position_norm[np.newaxis, :], ax=axs[0],
#         colorbar=False, legend=False
#     )

#     # 2. High-reverb without interference
#     RT60 = 1.5
#     signals_reverb = _simulate_3d(rt60=RT60, scenario=base_scenario)[-1]

#     estimated_positions, srp_map_reverb, candidate_grid = srp_func.forward(signals_reverb)

#     axs[1].set_title(f"(b) RT60 of {RT60}s")
#     plot_azimuth_elevation_grid(
#         candidate_grid, srp_map=srp_map_reverb,
#         source_positions=source_position_norm[np.newaxis, :], ax=axs[1],
#         colorbar=False, legend=False
#     )

#     # 3. Low-reverb with interference
#     signals = _simulate_3d(interferer=True, interferer_snr=0, scenario=base_scenario)[-1]

#     estimated_positions, srp_map, candidate_grid = srp_func.forward(signals)

#     axs[2].set_title("(c) Directional interference (0 dB SNR)")
#     plot_azimuth_elevation_grid(
#         candidate_grid, srp_map=srp_map,
#         source_positions=source_position_norm[np.newaxis, :],
#         interferer_positions=interferer_position_norm[np.newaxis, :], ax=axs[2],
#         colorbar=False
#         #output_path="tests/temp/interferer.png",
#     )
#     # Set y-ticks off
#     #axs[0].set_yticks([])
#     axs[0].set_ylabel("")
#     #axs[2].set_yticks([])
#     axs[2].set_ylabel("")

#     # Set x-ticks off
#     axs[0].set_xlabel("")
#     axs[1].set_xlabel("")
#     axs[0].set_xticks([])
#     axs[1].set_xticks([])

#     axs[1].yaxis.set_label_position("right")
#     #axs[1].yaxis.tick_right()

#     plt.tight_layout()
#     plt.savefig("tests/temp/srp_doa_az_el.pdf")



def test_doa_3d_clean():
    os.makedirs("tests/temp", exist_ok=True)

    # Create 3 scenarios: low-reverb without interference, high-reverb without interference and low-reverb with interference

    fig, ax = plt.subplots(figsize=(5, 3))

    # 1. Low-reverb without interference
    (
        fs, _,
        mic_positions_relative, _,
        source_position_norm, _,
        interferer_position_norm, _,
        signals,
    ) = base_scenario = _simulate_3d(interferer=False)

    srp_func = ConventionalSrp(fs, "doa_2D", 200,
        mic_positions=mic_positions_relative, interpolation=False,
        mode="gcc_phat_freq",
        n_average_samples=5, freq_cutoff_in_hz=None)

    estimated_positions, srp_map_base, candidate_grid = srp_func.forward(signals)

    ax.set_title(f"SRP map")
    plot_azimuth_elevation_grid(
        candidate_grid, srp_map=srp_map_base,
        source_positions=source_position_norm[np.newaxis, :], ax=ax,
        colorbar=True, legend=False
    )

    plt.tight_layout()
    plt.savefig("tests/temp/srp_doa_az_el_clean.png")


def _simulate_2d(rt60=0.3, interferer=False):
    fs = 16000
    (
        room_dims,
        mic_positions_norm,
        mic_positions,
        source_position_norm,
        source_position,
        interferer_position_norm,
        interferer_position
    ) = generate_random_scenario(mode="2d")

    # 1. Generate received source signal 
    source_signal = sf.read("tests/fixtures/p225_001.wav")[0]

    room = pra.ShoeBox(room_dims, fs=fs,)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(mic_positions.T)
        
    room.simulate()
    signals = room.mic_array.signals

    return fs, room_dims, mic_positions_norm, mic_positions, source_position_norm, source_position, signals


def _simulate_3d(rt60=0.3, interferer=False, interferer_snr=10, scenario=None):
    fs = 16000

    if scenario is not None:
        (
            room_dims,
            mic_positions_relative,
            mic_positions,
            source_position_norm,
            source_position,
            interferer_position_norm,
            interferer_position
        ) = scenario
    else:
        (
            room_dims,
            mic_positions_relative,
            mic_positions,
            source_position_norm,
            source_position,
            interferer_position_norm,
            interferer_position
        ) = generate_random_scenario(mode="3d", interferer=interferer)

    # 1. Generate received source signal
    source_signal = np.random.normal(size=fs)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(room_dims, fs=fs,
                       materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(mic_positions.T)
        
    room.simulate()
    signals = room.mic_array.signals[:, :fs]

    # 2. Generate received interferer signal
    # at the provided SNR
    if interferer:
        room = pra.ShoeBox(room_dims, fs=fs,
                       materials=pra.Material(e_absorption), max_order=max_order)
        room.add_microphone_array(mic_positions.T)
        interferer_signal = np.random.normal(size=fs)
        room.add_source(interferer_position, signal=interferer_signal)
        room.simulate()

        # Compute the scaling factor for the interferer
        # to achieve the desired SNR
        interferer_recorded = room.mic_array.signals[:, :fs]
        scaling_factor = np.sqrt(
            np.sum(signals ** 2)
            / np.sum(interferer_recorded ** 2)
        )
        
        # Scale the interferer signal to achieve the desired SNR
        interferer_recorded *= scaling_factor * 10 ** (-interferer_snr / 20)
        
        signals += interferer_recorded
    return (
        fs, room_dims, mic_positions_relative, mic_positions,
        source_position_norm, source_position,
        interferer_position_norm, interferer_position, signals
    )

def generate_random_scenario(mode="2d", interferer=False):
    interferer_position_norm = None
    interferer_position = None

    if mode == "2d":
        # Create a room with sources and microphones
        room_dims = [5, 5]

        # Place a microphone array in the room  
        mic_positions_norm = pra.circular_2D_array([0, 0], 8, 0, 0.05).T
        mic_positions = mic_positions_norm + np.array([2.5, 2.5])
        # Place the source 1 meter away from the microphone array at a random angle
        
        angle = np.random.uniform(low=0, high=2*np.pi)
        source_position_norm = np.array([np.cos(angle), np.sin(angle)])
        source_position = source_position_norm + np.array([2.5])

        if interferer:
            # Place the interferer 1 meter away from the microphone array at a random angle
            angle = np.random.uniform(low=0, high=2*np.pi)
            interferer_position_norm = np.array([np.cos(angle), np.sin(angle)])
            interferer_position = interferer_position_norm + np.array([2.5])

    elif mode == "3d":
        # Create a room with sources and microphones
        room_dims = [5, 5, 3]

        # Place a tetrahedral microphone array in the room
        mic_positions = np.array([
            [ 0.0243,  0.0243,  0.024],
            [ 0.0243, -0.0243, -0.024],
            [-0.0243,  0.0243, -0.024],
            [-0.0243, -0.0243,  0.024]
        ])
        mic_positions_relative = mic_positions.copy()
        mic_positions_norm = mic_positions / np.linalg.norm(mic_positions, axis=1, keepdims=True)

        mic_positions = mic_positions + np.array([2.5, 2.5, 1.5])
        # Place the source 1 meter away from the microphone array at a random angle
        
        azimuth = np.random.uniform(low=0, high=2*np.pi)
        elevation = np.random.uniform(low=0, high=np.pi)
        source_position_norm = np.array([
            np.cos(azimuth) * np.sin(elevation),
            np.sin(azimuth) * np.sin(elevation),
            np.cos(elevation)
        ])
        source_position = source_position_norm + np.array([2.5, 2.5, 1.5])

        if interferer:
            # Place the interferer 1 meter away from the microphone array at a random angle
            azimuth = np.random.uniform(low=0, high=2*np.pi)
            elevation = np.random.uniform(low=0, high=np.pi)
            interferer_position_norm = np.array([
                np.cos(azimuth) * np.sin(elevation),
                np.sin(azimuth) * np.sin(elevation),
                np.cos(elevation)
            ])
            interferer_position = interferer_position_norm + np.array([2.5, 2.5, 1.5])

    return (
            room_dims,
            mic_positions_relative,
            mic_positions,
            source_position_norm,
            source_position,
            interferer_position_norm,
            interferer_position
        )
