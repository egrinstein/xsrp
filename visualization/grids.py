import matplotlib.pyplot as plt
import numpy as np


def plot_uniform_cartesian_grid(
    grid,
    dims,
    output_path=None,
    srp_map=None,
    mic_positions=None,
    source_positions=None,
    ax=None,
):
    dims = np.array(dims)
    n_dims = len(dims)

    points = grid.asarray()

    points = np.array(points)

    show = False
    if ax is None:
        show = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if n_dims == 3 else None)

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")

    cmap = "viridis" if srp_map is not None else None
    scatter = ax.scatter(
        *[points[:, i] for i in range(n_dims)], c=srp_map, marker="o", cmap=cmap
    )

    # Set axes limits
    if dims[0].shape == 1:
        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        if len(dims) == 3:
            ax.set_zlim(0, dims[2])
    elif dims[0].shape == 2:
        ax.set_xlim(dims[0][0], dims[0][1])
        ax.set_ylim(dims[1][0], dims[1][1])
        if len(dims) == 3:
            ax.set_zlim(dims[2][0], dims[2][1])

    if srp_map is not None:
        plt.colorbar(scatter, label="Likelihood", ax=ax)

    if mic_positions is not None:
        ax.scatter(
            *[mic_positions[:, i] for i in range(n_dims)],
            marker="^",
            color="red",
            label="Mic. positions"
        )
        ax.legend()
    if source_positions is not None:
        ax.scatter(
            *[source_positions[:, i] for i in range(n_dims)],
            marker="x",
            color="orange",
            label="Source positions"
        )
        ax.legend()

    if output_path is None:
        if show:
            plt.show()
    else:
        plt.savefig(output_path)

    return ax


def plot_azimuth_elevation_grid(
    grid, output_path=None, srp_map=None, source_positions=None,
    interferer_positions=None, ax=None
):
    """
    Plot a grid of points located in the unit sphere. The points are
    defined by their cartesian coordinates, but they are plotted in the
    azimuth-elevation plane.
    """

    points = grid.asarray()

    # 1. Convert cartesian coordinates to spherical coordinates
    points_sph = _cart_to_sph(points)

    # 2. Plot the points in the azimuth-elevation plane
    show = False
    if ax is None:
        show = True
        fig, ax = plt.subplots()

    cmap = "viridis" if srp_map is not None else None
    scatter = ax.scatter(
        points_sph[:, 2], points_sph[:, 1], c=srp_map, marker="o", cmap=cmap
    )

    if srp_map is not None:
        plt.colorbar(scatter, label="Likelihood", ax=ax)

    if source_positions is not None:
        # Convert cartesian coordinates to spherical coordinates
        source_positions_sph = _cart_to_sph(source_positions)

        ax.scatter(
            source_positions_sph[:, 2],
            source_positions_sph[:, 1],
            marker="x",
            color="orange",
            label="Source positions",
        )
    if interferer_positions is not None:
        interferer_positions_sph = _cart_to_sph(interferer_positions)

        ax.scatter(
            interferer_positions_sph[:, 2],
            interferer_positions_sph[:, 1],
            marker="x",
            color="red",
            label="Interferer positions",
        )
    
    if source_positions is not None or interferer_positions is not None:
        ax.legend()

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")

    if output_path is None:
        if show:
            plt.show()
    else:
        plt.savefig(output_path)


def _cart_to_sph(points_cart: np.ndarray, degrees=True):
    """Convert an array of shape (n_points, 3) in Cartesian coordinates to spherical coordinates"""

    r = np.linalg.norm(points_cart, axis=1)
    theta = np.arccos(points_cart[:, 2] / r)
    phi = np.arctan2(points_cart[:, 1], points_cart[:, 0])

    sph = np.stack([r, theta, phi], axis=1)

    if degrees:
        sph[:, 1:] *= 180 / np.pi

    return sph
