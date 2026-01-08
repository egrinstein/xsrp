import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from xsrp.visualization.grids import (
    plot_azimuth_elevation_grid
)
from xsrp.visualization.polar import plot_polar_srp_map
from xsrp.xsrp import XSrp


def test_doa_2d_clean():
    """Test 2D DOA (azimuth only) with clean signal."""
    os.makedirs("tests/temp", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    # Create 2D DOA scenario
    fs = 16000
    room_dims = [5, 5, 3]
    
    # Create circular microphone array (2D, in xy plane)
    radius_mm = 35
    radius_m = radius_mm / 1000.0
    n_mics = 6
    mic_angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    mic_positions_relative = np.array([
        [radius_m * np.cos(angle), radius_m * np.sin(angle)]
        for angle in mic_angles
    ])  # 2D positions for 1D DOA
    
    # Place array center in room
    array_center = np.array([2.5, 2.5, 1.5])
    mic_positions_3d = np.column_stack([mic_positions_relative, np.zeros(n_mics)]) + array_center
    
    # Source direction (azimuth only) - place source 1 meter away from array center
    source_azimuth = np.random.uniform(low=0, high=2*np.pi)
    source_position_norm = np.array([np.cos(source_azimuth), np.sin(source_azimuth), 0.0])
    # Place source 1 meter away from array center
    source_position = array_center + source_position_norm * 1.0
    
    # Generate source signal
    source_signal = np.random.normal(size=fs)
    
    # Simulate room (low reverb for clean signal)
    rt60 = 0.2
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(room_dims, fs=fs,
                       materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(source_position, signal=source_signal)
    room.add_microphone_array(mic_positions_3d.T)
    room.simulate()
    signals = room.mic_array.signals[:, :fs]

    # Create SRP processor for 1D DOA (azimuth only)
    srp_func = XSrp(fs, "doa_1D", 360,
        mic_positions=mic_positions_relative, interpolation=False,
        mode="gcc_phat_freq",
        n_average_samples=5, freq_cutoff_in_hz=None)

    estimated_positions, srp_map, candidate_grid = srp_func.forward(signals)

    ax.set_title("SRP map (2D DOA - Azimuth only)", pad=20)
    plot_polar_srp_map(
        candidate_grid, srp_map=srp_map,
        tracked_position=source_position_norm, ax=ax,
        colorbar=False, show_tracked=True
    )

    plt.tight_layout()
    plt.savefig("tests/temp/srp_doa_2d_clean.png")


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

    srp_func = XSrp(fs, "doa_2D", 200,
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
