import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from visualization.polar import plot_polar_srp_map
from xsrp.xsrp import XSrp


def test_visualize_frequency_weighting_comparison():
    """Compare SRP maps using different frequency weighting methods."""
    os.makedirs("tests/temp", exist_ok=True)

    # Create a 2D DOA scenario with a single source
    fs = 16000
    
    # Create circular microphone array (2D, in xy plane)
    radius_mm = 35  # 35mm radius
    radius_m = radius_mm / 1000.0
    n_mics = 6
    mic_angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    mic_positions_2d = np.array([
        [radius_m * np.cos(angle), radius_m * np.sin(angle), 0.0]
        for angle in mic_angles
    ])
    
    # Source direction (azimuth only for 1D DOA)
    source_azimuth = np.pi / 4  # 45 degrees
    source_position_norm = np.array([np.cos(source_azimuth), np.sin(source_azimuth), 0.0])
    
    # Generate source signal
    source_signal = sf.read("tests/fixtures/p225_001.wav")[0]
    source_signal = source_signal[:fs]  # Use 1 second
    
    # Simulate room (far-field, so we can use a simple room)
    room_dims = [5, 5, 3]
    room = pra.ShoeBox(room_dims, fs=fs, max_order=0)  # No reverberation for clarity
    room.add_source([2.5, 2.5, 1.5], signal=source_signal)
    room.add_microphone_array(mic_positions_2d.T + np.array([[2.5], [2.5], [1.5]]))
    room.simulate()
    signals = room.mic_array.signals[:, :fs]
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    axs = axs.flatten()
    
    # Test all four methods
    methods = [
        (None, "No Weighting"),
        ("coherence", "Coherence Weighting"),
        ("sparsity", "Sparsity Weighting"),
        ("par", "PAR Weighting")
    ]
    
    n_azimuth_cells = 360  # 1 degree resolution
    
    for idx, (method, title) in enumerate(methods):
        ax = axs[idx]
        
        # Create SRP processor with frequency weighting
        srp_processor = XSrp(
            fs=fs,
            grid_type="doa_1D",
            n_grid_cells=n_azimuth_cells,
            mic_positions=mic_positions_2d[:, :2],  # 2D positions for 1D DOA
            mode="gcc_phat_freq",
            interpolation=False,
            n_average_samples=1,
            sharpening=1.0,
            frequency_weighting=method
        )
        
        # Compute SRP map
        estimated_positions, srp_map, candidate_grid = srp_processor.forward(signals)
        
        # Plot SRP map
        plot_polar_srp_map(
            candidate_grid,
            srp_map,
            tracked_position=source_position_norm,
            ax=ax,
            colorbar=False,
            show_tracked=True
        )
        
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    plt.savefig("tests/temp/frequency_weighting_comparison.png", dpi=150)
    plt.close()


def test_visualize_frequency_weighting_comparison_reverb():
    """Compare SRP maps with reverberation to show weighting benefits."""
    os.makedirs("tests/temp", exist_ok=True)

    # Create a 2D DOA scenario with reverberation
    fs = 16000
    rt60 = 0.3  # Moderate reverberation
    
    # Create circular microphone array (2D, in xy plane)
    radius_mm = 35
    radius_m = radius_mm / 1000.0
    n_mics = 6
    mic_angles = np.linspace(0, 2*np.pi, n_mics, endpoint=False)
    mic_positions_2d = np.array([
        [radius_m * np.cos(angle), radius_m * np.sin(angle), 0.0]
        for angle in mic_angles
    ])
    
    # Source direction
    source_azimuth = np.pi / 3  # 60 degrees
    source_position_norm = np.array([np.cos(source_azimuth), np.sin(source_azimuth), 0.0])
    
    # Generate source signal
    source_signal = sf.read("tests/fixtures/p225_001.wav")[0]
    source_signal = source_signal[:fs]
    
    # Simulate room with reverberation
    room_dims = [5, 5, 3]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dims)
    room = pra.ShoeBox(room_dims, fs=fs,
                       materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source([2.5, 2.5, 1.5], signal=source_signal)
    room.add_microphone_array(mic_positions_2d.T + np.array([[2.5], [2.5], [1.5]]))
    room.simulate()
    signals = room.mic_array.signals[:, :fs]
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='polar'))
    axs = axs.flatten()
    
    # Test all four methods
    methods = [
        (None, "No Weighting"),
        ("coherence", "Coherence Weighting"),
        ("sparsity", "Sparsity Weighting"),
        ("par", "PAR Weighting")
    ]
    
    n_azimuth_cells = 360
    
    for idx, (method, title) in enumerate(methods):
        ax = axs[idx]
        
        # Create SRP processor with frequency weighting
        srp_processor = XSrp(
            fs=fs,
            grid_type="doa_1D",
            n_grid_cells=n_azimuth_cells,
            mic_positions=mic_positions_2d[:, :2],
            mode="gcc_phat_freq",
            interpolation=False,
            n_average_samples=1,
            sharpening=1.0,
            frequency_weighting=method
        )
        
        # Compute SRP map
        estimated_positions, srp_map, candidate_grid = srp_processor.forward(signals)
        
        # Plot SRP map
        plot_polar_srp_map(
            candidate_grid,
            srp_map,
            tracked_position=source_position_norm,
            ax=ax,
            colorbar=False,
            show_tracked=True
        )
        
        ax.set_title(title, pad=20)
    
    plt.tight_layout()
    plt.savefig("tests/temp/frequency_weighting_comparison_reverb.png", dpi=150)
    plt.close()

