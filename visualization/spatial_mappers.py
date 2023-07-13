import matplotlib.pyplot as plt
import numpy as np

from spatial_mappers import temporal_spatial_mapper


def plot_temporal_spatial_mapper(grid, mapper, microphone_positions=None, output_path=None):
    """Plot the temporal spatial mapper.

    Parameters
    ----------
    grid : np.array (n_points, n_dimensions)
        The grid of candidate positions associated to a theoretical TDOA value.
    mapper : np.array (n_points, n_mics, n_mics)
        The temporal spatial mapper to plot. The mapper can is of shape 3D (n_points, n_mics, n_mics)
    microphone_positions : np.array (n_microphones, n_dimensions)
        The positions of the microphones.
    output_path : str
        The path to save the plot to. If None, the plot is shown instead of saved.
    """

    if mapper.ndim != 3:
        raise ValueError("The temporal spatial mapper must be 4-dimensional.")
    
    n_mics = mapper.shape[-1]
    n_dims = grid.shape[1]
    n_pairs = int(n_mics * (n_mics - 1) / 2)

    fig, axs = plt.subplots(n_pairs, figsize=(10, 10))

    if n_pairs == 1:
        axs = np.expand_dims(axs, axis=0)

    n_pair = 0
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            if len(n_dims) == 2:
                ax = fig.add_subplot(111)
                axs[n_pair].scatter(grid[:, 0], grid[:, 1], c=mapper[:, i, j], marker='o', cmap='viridis')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')

            elif len(n_dims) == 3:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2], c=mapper[:, i, j], marker='o', cmap='viridis')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

            axs[n_pair].scatter(grid[:, 0], grid[:, 1], c=mapper[:, i, j], marker='o', cmap='viridis')
            axs[n_pair].set_title("TDOA between microphones {} and {}".format(i, j))
            axs[n_pair].set_xlabel("X")
            axs[n_pair].set_ylabel("Y")
            n_pair += 1
        
            if microphone_positions is not None:
                axs[n_pair].scatter(microphone_positions[i], microphone_positions[j], c='red', marker='x', label="Mic. positions")


    fig.colorbar(label="TDOA", mappable=plt.cm.ScalarMappable(cmap='viridis'))
    
    plt.legend()
    plt.tight_layout()


    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
