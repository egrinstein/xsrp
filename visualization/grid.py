import matplotlib.pyplot as plt
import numpy as np


def plot_uniform_cartesian_grid(grid, dims, output_path=None):
    dims = np.array(dims)

    points = []
    likelihoods = []
    for cell in grid:
        points.append(cell.position)
        likelihoods.append(cell.likelihood)

    points = np.array(points)

    fig = plt.figure()
    if len(dims) == 2:
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], c=likelihoods, marker='o', cmap='viridis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    elif len(dims) == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=likelihoods, marker='o', cmap='viridis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

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

    fig.colorbar(label="Likelihood", mappable=plt.cm.ScalarMappable(cmap='viridis'))

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
