import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

from xsrp.grids import UniformSphericalGrid
from xsrp.visualization.grids import plot_azimuth_elevation_grid
from xsrp.visualization.polar import plot_polar_srp_map


def test_azimuth_limits_1d_doa():
    """Test azimuth limits for 1D DOA grid (azimuth only)."""
    # Create grid with azimuth range from pi/4 to 3*pi/4 (90 degrees)
    grid = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=np.pi/4,
        azimuth_max=3*np.pi/4
    )
    
    # Verify grid has correct shape
    assert grid.grid_shape[0] < 360  # Should have fewer cells than full range
    assert grid.grid_shape[1] == 2  # 2D positions for 1D DOA
    
    # Verify all azimuths are within the specified range
    # Check the stored azimuth_range directly (more reliable than computing from positions)
    assert np.all(grid.azimuth_range >= np.pi/4)
    assert np.all(grid.azimuth_range < 3*np.pi/4)
    
    # Also verify positions match
    positions = grid.asarray()
    computed_azimuths = np.arctan2(positions[:, 1], positions[:, 0])
    # Normalize azimuths to [0, 2*pi] to match grid's convention
    computed_azimuths = np.where(computed_azimuths < 0, computed_azimuths + 2*np.pi, computed_azimuths)
    
    # Allow small tolerance for floating point errors
    assert np.allclose(computed_azimuths, grid.azimuth_range, atol=1e-10)
    
    # Verify grid length
    assert len(grid) == grid.grid_shape[0]


def test_azimuth_limits_exclude_back():
    """Test excluding the back of the array (azimuth from pi/2 to 3*pi/2)."""
    # Exclude back half (azimuth from pi/2 to 3*pi/2)
    # Keep front half (azimuth from 3*pi/2 to 2*pi and 0 to pi/2)
    # Since we can't wrap around, we'll test two separate ranges
    # Test 1: Front right quadrant (0 to pi/2)
    grid1 = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=0,
        azimuth_max=np.pi/2
    )
    
    assert np.all(grid1.azimuth_range >= 0)
    assert np.all(grid1.azimuth_range < np.pi/2)
    
    # Test 2: Front left quadrant (3*pi/2 to 2*pi)
    grid2 = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=3*np.pi/2,
        azimuth_max=2*np.pi
    )
    
    assert np.all(grid2.azimuth_range >= 3*np.pi/2)
    assert np.all(grid2.azimuth_range < 2*np.pi)


def test_elevation_limits_2d_doa():
    """Test elevation limits for 2D DOA grid."""
    # Create grid with elevation range from pi/6 to pi/2 (30 to 90 degrees)
    grid = UniformSphericalGrid(
        n_azimuth_cells=180,
        n_elevation_cells=90,
        elevation_min=np.pi/6,
        elevation_max=np.pi/2
    )
    
    # Verify grid has correct shape
    assert grid.grid_shape[0] == 180  # All azimuths included
    assert grid.grid_shape[1] < 90  # Fewer elevation cells
    assert grid.grid_shape[2] == 3  # 3D positions for 2D DOA
    
    # Verify all elevations are within the specified range
    positions = grid.asarray()
    # Convert cartesian to spherical coordinates
    # elevation = arccos(z) for unit vectors
    elevations = np.arccos(positions[:, 2])
    
    assert np.all(elevations >= np.pi/6)
    assert np.all(elevations < np.pi/2)
    
    # Verify grid length
    assert len(grid) == grid.grid_shape[0] * grid.grid_shape[1]


def test_combined_azimuth_elevation_limits():
    """Test combined azimuth and elevation limits."""
    grid = UniformSphericalGrid(
        n_azimuth_cells=360,
        n_elevation_cells=180,
        azimuth_min=np.pi/4,
        azimuth_max=3*np.pi/4,
        elevation_min=np.pi/6,
        elevation_max=np.pi/3
    )
    
    # Check azimuths using stored range
    assert np.all(grid.azimuth_range >= np.pi/4)
    assert np.all(grid.azimuth_range < 3*np.pi/4)
    
    # Check elevations using stored range
    assert np.all(grid.elevation_range >= np.pi/6)
    assert np.all(grid.elevation_range < np.pi/3)
    
    # Verify positions match
    positions = grid.asarray()
    
    # Check elevations from positions
    elevations = np.arccos(positions[:, 2])
    # Elevations should match the stored range (repeated for each azimuth)
    unique_elevations = np.unique(elevations)
    assert np.allclose(np.sort(unique_elevations), np.sort(grid.elevation_range), atol=1e-10)


def test_default_behavior_no_limits():
    """Test that default behavior (no limits) matches original behavior."""
    grid_with_limits = UniformSphericalGrid(
        n_azimuth_cells=180,
        azimuth_min=0,
        azimuth_max=2*np.pi
    )
    
    grid_without_limits = UniformSphericalGrid(
        n_azimuth_cells=180
    )
    
    # Both should have the same number of cells
    assert len(grid_with_limits) == len(grid_without_limits)
    assert grid_with_limits.grid_shape[0] == grid_without_limits.grid_shape[0]
    
    # Positions should be identical
    positions_with_limits = grid_with_limits.asarray()
    positions_without_limits = grid_without_limits.asarray()
    
    np.testing.assert_array_almost_equal(positions_with_limits, positions_without_limits)


def test_invalid_azimuth_range():
    """Test that invalid azimuth ranges raise errors."""
    # min >= max
    with pytest.raises(ValueError, match="azimuth_min.*must be less than azimuth_max"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            azimuth_min=np.pi,
            azimuth_max=np.pi/2
        )
    
    # min < 0
    with pytest.raises(ValueError, match="Azimuth range must be within"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            azimuth_min=-0.1,
            azimuth_max=np.pi
        )
    
    # max > 2*pi
    with pytest.raises(ValueError, match="Azimuth range must be within"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            azimuth_min=0,
            azimuth_max=2*np.pi + 0.1
        )


def test_invalid_elevation_range():
    """Test that invalid elevation ranges raise errors."""
    # min >= max
    with pytest.raises(ValueError, match="elevation_min.*must be less than elevation_max"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            n_elevation_cells=90,
            elevation_min=np.pi/2,
            elevation_max=np.pi/4
        )
    
    # min < 0
    with pytest.raises(ValueError, match="Elevation range must be within"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            n_elevation_cells=90,
            elevation_min=-0.1,
            elevation_max=np.pi/2
        )
    
    # max > pi
    with pytest.raises(ValueError, match="Elevation range must be within"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            n_elevation_cells=90,
            elevation_min=0,
            elevation_max=np.pi + 0.1
        )


def test_empty_azimuth_range():
    """Test that empty azimuth range raises error."""
    # Range that doesn't include any cells
    with pytest.raises(ValueError, match="No azimuth cells in the specified range"):
        UniformSphericalGrid(
            n_azimuth_cells=10,
            azimuth_min=2*np.pi - 0.1,
            azimuth_max=2*np.pi - 0.05
        )


def test_empty_elevation_range():
    """Test that empty elevation range raises error."""
    # Range that doesn't include any cells
    with pytest.raises(ValueError, match="No elevation cells in the specified range"):
        UniformSphericalGrid(
            n_azimuth_cells=180,
            n_elevation_cells=10,
            elevation_min=np.pi - 0.1,
            elevation_max=np.pi - 0.05
        )


def test_azimuth_limits_with_srp():
    """Test that azimuth-limited grid works with SRP processing."""
    # Create a simple test scenario
    n_mics = 4
    mic_positions = np.array([
        [0.01, 0.0],
        [0.0, 0.01],
        [-0.01, 0.0],
        [0.0, -0.01]
    ])
    
    # Create grid with limited azimuth range (front right quadrant only)
    grid = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=0,
        azimuth_max=np.pi/2  # Front right quadrant
    )
    
    # The grid should work with the spatial mapper
    from xsrp.spatial_mappers import tdoa_mapper
    mapper = tdoa_mapper(grid, mic_positions)
    
    # Verify mapper shape matches grid
    assert mapper.shape[0] == len(grid)
    assert mapper.shape[1] == n_mics
    assert mapper.shape[2] == n_mics


def test_elevation_limits_with_srp():
    """Test that elevation-limited grid works with SRP processing."""
    from xsrp.spatial_mappers import tdoa_mapper
    
    # Create a simple test scenario
    n_mics = 4
    mic_positions = np.array([
        [0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, -0.01, 0.0]
    ])
    
    # Create grid with limited elevation range (horizontal plane only)
    grid = UniformSphericalGrid(
        n_azimuth_cells=180,
        n_elevation_cells=90,
        elevation_min=np.pi/2 - np.pi/6,  # 60 degrees
        elevation_max=np.pi/2 + np.pi/6   # 120 degrees (but max is pi, so effectively pi/2 to pi/2+pi/6)
    )
    
    # The grid should work with the spatial mapper
    mapper = tdoa_mapper(grid, mic_positions)
    
    # Verify mapper shape matches grid
    assert mapper.shape[0] == len(grid)
    assert mapper.shape[1] == n_mics
    assert mapper.shape[2] == n_mics


def test_visualize_azimuth_limits_1d():
    """Visualize 1D DOA grid with azimuth limits."""
    os.makedirs("tests/temp", exist_ok=True)
    
    # Create grid with limited azimuth range (front half only)
    grid = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=3*np.pi/2,
        azimuth_max=2*np.pi
    )
    
    # Create a dummy SRP map for visualization
    srp_map = np.random.rand(len(grid))
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.set_title("Azimuth-Limited Grid (Front Half Only)", pad=20)
    plot_polar_srp_map(grid, srp_map, ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig("tests/temp/grid_azimuth_limits_1d.png")
    plt.close()


def test_visualize_elevation_limits_2d():
    """Visualize 2D DOA grid with elevation limits."""
    os.makedirs("tests/temp", exist_ok=True)
    
    # Create grid with limited elevation range (horizontal plane)
    grid = UniformSphericalGrid(
        n_azimuth_cells=180,
        n_elevation_cells=90,
        elevation_min=np.pi/2 - np.pi/6,  # 60 degrees
        elevation_max=np.pi/2 + np.pi/6   # 120 degrees
    )
    
    # Create a dummy SRP map for visualization
    srp_map = np.random.rand(len(grid))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Elevation-Limited Grid (Horizontal Plane)", pad=20)
    plot_azimuth_elevation_grid(grid, srp_map=srp_map, ax=ax, colorbar=True)
    plt.tight_layout()
    plt.savefig("tests/temp/grid_elevation_limits_2d.png")
    plt.close()


def test_visualize_combined_limits():
    """Visualize 2D DOA grid with combined azimuth and elevation limits."""
    os.makedirs("tests/temp", exist_ok=True)
    
    # Create grid with both azimuth and elevation limits
    grid = UniformSphericalGrid(
        n_azimuth_cells=360,
        n_elevation_cells=180,
        azimuth_min=np.pi/4,
        azimuth_max=3*np.pi/4,
        elevation_min=np.pi/6,
        elevation_max=np.pi/3
    )
    
    # Create a dummy SRP map for visualization
    srp_map = np.random.rand(len(grid))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Combined Azimuth and Elevation Limits", pad=20)
    plot_azimuth_elevation_grid(grid, srp_map=srp_map, ax=ax, colorbar=True)
    plt.tight_layout()
    plt.savefig("tests/temp/grid_combined_limits_2d.png")
    plt.close()


def test_visualize_comparison_full_vs_limited():
    """Compare full grid vs limited grid side by side."""
    os.makedirs("tests/temp", exist_ok=True)
    
    # Create full grid
    grid_full = UniformSphericalGrid(n_azimuth_cells=360)
    
    # Create limited grid (front half only)
    grid_limited = UniformSphericalGrid(
        n_azimuth_cells=360,
        azimuth_min=3*np.pi/2,
        azimuth_max=2*np.pi
    )
    
    # Create dummy SRP maps
    srp_map_full = np.random.rand(len(grid_full))
    srp_map_limited = np.random.rand(len(grid_limited))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection='polar'))
    
    ax1.set_title("Full Grid (360°)", pad=20)
    plot_polar_srp_map(grid_full, srp_map_full, ax=ax1, colorbar=False)
    
    ax2.set_title("Limited Grid (270°-360°)", pad=20)
    plot_polar_srp_map(grid_limited, srp_map_limited, ax=ax2, colorbar=False)
    
    plt.tight_layout()
    plt.savefig("tests/temp/grid_comparison_full_vs_limited.png")
    plt.close()

