import matplotlib.pyplot as plt
import numpy as np
from xsrp.grids import UniformSphericalGrid


def plot_polar_srp_map(
    grid: UniformSphericalGrid,
    srp_map: np.ndarray,
    tracked_position: np.ndarray = None,
    ax=None,
    colorbar=True,
    radial_max=None,
    show_tracked=True
):
    """Plot SRP map in polar coordinates (azimuth only).
    
    Parameters
    ----------
    grid : UniformSphericalGrid
        The grid used for SRP computation (should be 1D DOA grid)
    srp_map : np.ndarray
        SRP map values corresponding to grid points
    tracked_position : np.ndarray, optional
        Tracked source position (cartesian unit vector) to display as marker
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes with polar projection. If None, creates new figure.
    colorbar : bool, optional
        Whether to show colorbar. Defaults to True.
    radial_max : float, optional
        Maximum radial limit. If None, uses max(srp_map) * 1.1. Defaults to None.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with polar projection
    """
    if grid.n_elevation_cells != 0:
        raise ValueError("plot_polar_srp_map only supports 1D DOA (azimuth only) grids")
    
    show = False
    if ax is None:
        show = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    
    # Get azimuth angles from grid
    azimuths = grid.azimuth_range
    
    # Determine radial max limit
    if radial_max is None:
        srp_max = np.max(srp_map)
        radial_max = srp_max * 1.1 if srp_max > 0 else 1.0
        
    # Set fixed radial limits before plotting to prevent auto-scaling
    ax.set_ylim(0, radial_max)
        
    # Use a nice blue color for the plot
    line_color = '#1f77b4'  # Matplotlib default blue color
    line = ax.plot(azimuths, srp_map, linewidth=2, color=line_color)[0]
    ax.fill(azimuths, srp_map, alpha=0.3, color=line_color)
    
    # Set theta zero to top (north) and direction to clockwise
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xlabel("Azimuth (degrees)", labelpad=20)
    
    # Plot tracked position if provided and enabled
    if tracked_position is not None and show_tracked:
        # Convert cartesian unit vector to azimuth
        azimuth_tracked = np.arctan2(tracked_position[1], tracked_position[0])
        # Find corresponding SRP value
        grid_positions = grid.asarray()
        # Handle dimension mismatch: grid is 2D for 1D DOA, but tracked_position might be 3D
        tracked_pos_2d = tracked_position[:grid_positions.shape[1]]
        # Find closest grid point
        distances = np.linalg.norm(grid_positions - tracked_pos_2d, axis=1)
        closest_idx = np.argmin(distances)
        srp_value = srp_map[closest_idx]
        
        # Draw arrow from origin to estimated direction
        ax.annotate('', 
                   xy=(azimuth_tracked, srp_value),  # Arrow tip at estimated position
                   xytext=(azimuth_tracked, 0),      # Arrow start at origin
                   arrowprops=dict(arrowstyle='->', 
                                 color=line_color, 
                                 lw=2.5))
        
        # Add angle value text right above the arrow tip
        azimuth_deg = np.degrees(azimuth_tracked)
        # Normalize to 0-360 range for display
        if azimuth_deg < 0:
            azimuth_deg += 360
        # Position text slightly above the arrow tip
        text_radius = srp_value * 1.1  # 10% above the arrow tip
        ax.text(azimuth_tracked, text_radius, f'{azimuth_deg:.1f}°',
               color=line_color, fontsize=10, fontweight='bold',
               ha='center', va='bottom')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def update_polar_srp_map(
    ax,
    grid: UniformSphericalGrid,
    srp_map: np.ndarray,
    tracked_position: np.ndarray = None,
    radial_max=None,
    show_tracked=True
):
    """Update an existing polar plot with new SRP map data.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes with polar projection (from previous plot_polar_srp_map call)
    grid : UniformSphericalGrid
        The grid used for SRP computation
    srp_map : np.ndarray
        New SRP map values
    tracked_position : np.ndarray, optional
        New tracked source position
    radial_max : float, optional
        Maximum radial limit. If None, preserves existing limit or uses max(srp_map) * 1.1.
    show_tracked : bool, optional
        Whether to show the tracked position marker. Defaults to True.
    """
    # Get current radial max if not provided
    if radial_max is None:
        current_ylim = ax.get_ylim()
        radial_max = current_ylim[1] if current_ylim[1] > 0 else np.max(srp_map) * 1.1
    
    # Clear existing plot
    ax.clear()
    
    # Only show tracked position if requested
    tracked_pos = tracked_position if show_tracked else None
    
    # Replot with new data, preserving radial limits
    plot_polar_srp_map(grid, srp_map, tracked_pos, ax=ax, colorbar=False, radial_max=radial_max)
    
    # Redraw
    ax.figure.canvas.draw()

