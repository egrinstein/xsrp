import numpy as np
import os

from visualization.spatial_mappers import (
    plot_temporal_spatial_mapper,
)
from grids import (
    UniformCartesianGrid,
)
from spatial_mappers import temporal_spatial_mapper


def test_plot_spatial_mapper_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10), 5)

    microphone_positions = np.array([[2.5, 2.5], [7.5, 7.5]])

    mapper = temporal_spatial_mapper(grid, microphone_positions)

    plot_temporal_spatial_mapper(grid, mapper,
                                 microphone_positions,
                                 output_path="tests/temp/plot_spatial_mapper_2d.png")
