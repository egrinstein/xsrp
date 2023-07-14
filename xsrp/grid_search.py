import numpy as np

from .grids import Grid


def argmax_grid_searcher(grid: Grid, srp_map: np.array) -> set[np.array]:
    positions = grid.asarray()

    max_idx = np.argmax(srp_map)
    max_position = positions[max_idx]

    return max_position
