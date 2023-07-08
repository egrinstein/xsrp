import matplotlib.pyplot as plt
import numpy as np


def plot_uniform_cartesian_grid_3d(grid, dims, output_path=None):
    x = []
    y = []
    for cell in grid:
        x.append(cell.position)

    x = np.array(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])
    ax.set_zlim(0, dims[2])

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def plot_uniform_cartesian_grid_2d(grid, dims, output_path=None):
    x = []
    y = []
    for cell in grid:
        x.append(cell.position[0])
        y.append(cell.position[1])

    x = np.array(x)
    y = np.array(y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.set_xlim(0, dims[0])
    ax.set_ylim(0, dims[1])

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)




