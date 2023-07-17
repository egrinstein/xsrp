import matplotlib.pyplot as plt
import numpy as np
import warnings


def plot_uniform_cartesian_grid(grid, dims, output_path=None, srp_map=None,
                                mic_positions=None, source_positions=None):
    dims = np.array(dims)
    n_dims = len(dims)

    points = grid.asarray()

    points = np.array(points)

    fig = plt.figure()
    if n_dims == 2:
        ax = fig.add_subplot(111)
    elif n_dims == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("Height (m)")

    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Length (m)")

    cmap = "viridis" if srp_map is not None else None
    scatter = ax.scatter(*[points[:,i] for i in range(n_dims)], c=srp_map, marker="o", cmap=cmap)

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
        ax.scatter(*[mic_positions[:,i] for i in range(n_dims)], marker="^", color="red", label="Mic. positions")
        ax.legend()
    if source_positions is not None:
        ax.scatter(*[source_positions[:,i] for i in range(n_dims)], marker="x", color="orange", label="Source positions")
        ax.legend()
    

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
