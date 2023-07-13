import os

from visualization.grid import (
    plot_uniform_cartesian_grid,
    #plot_uniform_angular_grid
)
from grids import (
    UniformCartesianGrid,
    UniformAngularGrid
)


def test_plot_uniform_cartesian_grid_3d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10, 10), 5)
    plot_uniform_cartesian_grid(grid, (10, 10, 10), output_path="tests/temp/plot_grid_3d.png")


def test_plot_uniform_cartesian_grid_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10), 20)
    plot_uniform_cartesian_grid(grid, (10, 10), output_path="tests/temp/plot_grid_2d.png")


def test_plot_uniform_angular_grid_1d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformAngularGrid(90)
    plot_uniform_cartesian_grid(grid, [[-1, 1], [-1, 1]], output_path="tests/temp/plot_angular_grid_1d.png")


def test_plot_uniform_angular_grid_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformAngularGrid(30, 10)
    plot_uniform_cartesian_grid(grid, [[-1, 1], [-1, 1], [-1, 1]], output_path="tests/temp/plot_angular_grid_2d.png")
