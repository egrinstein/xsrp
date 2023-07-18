import numpy as np
import os

from visualization.spatial_mappers import (
    plot_pairwise_mapper,
)
from xsrp.grids import (
    UniformCartesianGrid,
)
from xsrp.spatial_mappers import (
    tdoa_mapper,
    integer_sample_mapper,
    fractional_sample_mapper
)

def test_plot_tdoa_mapper_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10), 100)

    microphone_positions = np.array([[2.5, 2.5], [7.5, 7.5]])

    mapper = tdoa_mapper(grid, microphone_positions)

    plot_pairwise_mapper(grid, mapper,
                                 microphone_positions,
                                 output_path="tests/temp/tdoa_mapper_2d.png")


def test_plot_integer_sample_mapper_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10), 100)

    microphone_positions = np.array([[2.5, 2.5], [2.6, 2.6]])

    mapper = integer_sample_mapper(grid, microphone_positions, 16000)

    plot_pairwise_mapper(grid, mapper,
                     microphone_positions,
                     output_path="tests/temp/integer_sample_mapper_2d.png",
                     unit="samples")


def test_plot_fractional_sample_mapper_2d():
    os.makedirs("tests/temp", exist_ok=True)

    grid = UniformCartesianGrid((10, 10), 100)

    microphone_positions = np.array([[2.5, 2.5], [2.6, 2.6]])

    mapper = fractional_sample_mapper(grid, microphone_positions, 16000)

    plot_pairwise_mapper(grid, mapper,
                     microphone_positions,
                     output_path="tests/temp/fractional_sample_mapper_2d.png",
                     unit="samples")
