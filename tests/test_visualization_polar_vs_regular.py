import matplotlib.pyplot as plt
import numpy as np
import os
import pyroomacoustics as pra
import soundfile as sf

from visualization.polar import plot_polar_srp_map
from xsrp.xsrp import XSrp


def plot_regular_srp_map(
    grid,
    srp_map: np.ndarray,
    tracked_position: np.ndarray = None,
    ax=None,
    show_tracked=True
):
    """Plot SRP map in regular cartesian coordinates (azimuth vs SRP value).
    
    Parameters
    ----------
    grid : UniformSphericalGrid
        The grid used for SRP computation (should be 1D DOA grid)
    srp_map : np.ndarray
        SRP map values corresponding to grid points
    tracked_position : np.ndarray, optional
        Tracked source position (cartesian unit vector) to display as marker
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes. If None, creates new figure.
    show_tracked : bool, optional
        Whether to show the tracked position marker. Defaults to True.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object
    """
    if grid.n_elevation_cells != 0:
        raise ValueError("plot_regular_srp_map only supports 1D DOA (azimuth only) grids")
    
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots()
    
    # Get azimuth angles from grid (convert to degrees)
    azimuths = grid.azimuth_range * 180 / np.pi
    
    # Plot SRP map
    line_color = '#1f77b4'
    ax.plot(azimuths, srp_map, linewidth=2, color=line_color, label='SRP map')
    ax.fill_between(azimuths, srp_map, alpha=0.3, color=line_color)
    
    # Plot tracked position if provided and enabled
    if tracked_position is not None and show_tracked:
        # Convert cartesian unit vector to azimuth
        azimuth_tracked = np.arctan2(tracked_position[1], tracked_position[0]) * 180 / np.pi
        # Find corresponding SRP value
        grid_positions = grid.asarray()
        tracked_pos_2d = tracked_position[:grid_positions.shape[1]]
        distances = np.linalg.norm(grid_positions - tracked_pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        srp_value = srp_map[closest_idx]
        
        ax.plot(azimuth_tracked, srp_value, 'o', color=line_color,
                markersize=10, markeredgecolor='white', markeredgewidth=1.5,
                label='Tracked position')
    
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("SRP Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def test_visualize_polar_vs_regular():
    """Compare polar and regular visualizations side-by-side."""
    os.makedirs("tests/temp", exist_ok=True)

    # Create a 1D DOA scenario with a single source
    fs = 16000
    
    # Create circular microphone array (2D, in xy plane)
    radius_mm = 35
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
    
    # Create SRP processor
    n_azimuth_cells = 360  # 1 degree resolution
    srp_processor = XSrp(
        fs=fs,
        grid_type="doa_1D",
        n_grid_cells=n_azimuth_cells,
        mic_positions=mic_positions_2d[:, :2],  # 2D positions for 1D DOA
        mode="gcc_phat_freq",
        interpolation=False,
        n_average_samples=1,
        sharpening=1.0,
        frequency_weighting=None
    )
    
    # Compute SRP map
    estimated_positions, srp_map, candidate_grid = srp_processor.forward(signals)
    
    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot polar visualization
    ax_polar = plt.subplot(121, projection='polar')
    plot_polar_srp_map(
        candidate_grid,
        srp_map,
        tracked_position=source_position_norm,
        ax=ax_polar,
        colorbar=False,
        show_tracked=True
    )
    ax_polar.set_title("Polar Visualization", pad=20, fontsize=14)
    
    # Plot regular visualization
    ax_regular = plt.subplot(122)
    plot_regular_srp_map(
        candidate_grid,
        srp_map,
        tracked_position=source_position_norm,
        ax=ax_regular,
        show_tracked=True
    )
    ax_regular.set_title("Regular (Cartesian) Visualization", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("tests/temp/polar_vs_regular_comparison.png", dpi=150)
    plt.close()


def test_visualize_polar_vs_regular_with_weighting():
    """Compare polar and regular visualizations with different weighting methods."""
    os.makedirs("tests/temp", exist_ok=True)

    # Create a 1D DOA scenario with reverberation
    fs = 16000
    rt60 = 0.3
    
    # Create circular microphone array
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
    
    # Test with coherence weighting
    n_azimuth_cells = 360
    srp_processor = XSrp(
        fs=fs,
        grid_type="doa_1D",
        n_grid_cells=n_azimuth_cells,
        mic_positions=mic_positions_2d[:, :2],
        mode="gcc_phat_freq",
        interpolation=False,
        n_average_samples=1,
        sharpening=1.0,
        frequency_weighting="coherence"
    )
    
    # Compute SRP map
    estimated_positions, srp_map, candidate_grid = srp_processor.forward(signals)
    
    # Create figure with 1x2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot polar visualization
    ax_polar = plt.subplot(121, projection='polar')
    plot_polar_srp_map(
        candidate_grid,
        srp_map,
        tracked_position=source_position_norm,
        ax=ax_polar,
        colorbar=False,
        show_tracked=True
    )
    ax_polar.set_title("Polar Visualization (Coherence Weighting)", pad=20, fontsize=14)
    
    # Plot regular visualization
    ax_regular = plt.subplot(122)
    plot_regular_srp_map(
        candidate_grid,
        srp_map,
        tracked_position=source_position_norm,
        ax=ax_regular,
        show_tracked=True
    )
    ax_regular.set_title("Regular Visualization (Coherence Weighting)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("tests/temp/polar_vs_regular_with_weighting.png", dpi=150)
    plt.close()

