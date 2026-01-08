import matplotlib.pyplot as plt
import numpy as np

from xsrp.grids import Grid


def plot_pairwise_mapper(grid: Grid, mapper, microphone_positions=None, output_path=None, unit="s"):
    """Plot a spatial mapper.

    Parameters
    ----------
    grid : Grid (n_points, n_dimensions)
        The grid of candidate positions
    mapper : np.array (n_points, n_mics, n_mics)
        The spatial mapper, where each point is associated to a grid cell.
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones. Optional.
    output_path : str
        The path to save the plot to. If None, the plot is shown instead of saved.
    """

    grid = grid.asarray()

    if unit == "s":
        mapper *= 1000 # convert to ms
        unit = "ms"

    if mapper.ndim != 3:
        raise ValueError("The spatial mapper must be 3-dimensional.")
    
    n_mics = mapper.shape[-1]
    n_dims = grid.shape[1]
    n_pairs = int(n_mics * (n_mics - 1) / 2)

    fig, axs = plt.subplots(n_pairs, figsize=(10, 10))

    if n_pairs == 1:
        axs = np.expand_dims(axs, axis=0)

    n_pair = 0
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            if n_dims == 2:
                axs[n_pair].scatter(grid[:, 0], grid[:, 1], c=mapper[:, i, j], marker="o", cmap="viridis")
                axs[n_pair].set_xlabel("Length (m)")
                axs[n_pair].set_ylabel("Width (m)")

            elif n_dims == 3:
                axs[n_pair] = fig.add_subplot(111, projection="3d")
                axs[n_pair].scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=mapper[:, i, j], marker="o", cmap="viridis")

                axs[n_pair].set_xlabel("Length (m)")
                axs[n_pair].set_ylabel("Width (m)")
                axs[n_pair].set_zlabel("Height (m)")

            scatter = axs[n_pair].scatter(grid[:, 0], grid[:, 1], c=mapper[:, i, j], marker="o", cmap="viridis")
            axs[n_pair].set_title("Spatial mapper for microphones {} and {}".format(i, j))
        
            if microphone_positions is not None:
                mic_pos_ij = np.array([microphone_positions[i], microphone_positions[j]])
                axs[n_pair].scatter(mic_pos_ij[:, 0], mic_pos_ij[:, 1], c="red", label="Mic. positions",
                                    marker="^")

            plt.colorbar(scatter, label=f"value ({unit})", ax=axs[n_pair])
            n_pair += 1
    
    plt.legend()
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
