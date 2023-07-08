import os

from visualization import (
    plot_uniform_cartesian_grid_2d,
    plot_uniform_cartesian_grid_3d
)
from grids import (
    create_uniform_cartesian_grid_2d,
    create_uniform_cartesian_grid_3d
)


def test_plot_uniform_cartesian_grid_3d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = create_uniform_cartesian_grid_3d((10, 10, 10), 5)
    plot_uniform_cartesian_grid_3d(grid, (10, 10, 10), output_path="tests/temp/plot_grid_3d.png")

def test_plot_uniform_cartesian_grid_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = create_uniform_cartesian_grid_2d((10, 10), 5)
    plot_uniform_cartesian_grid_2d(grid, (10, 10), output_path="tests/temp/plot_grid_2d.png")
