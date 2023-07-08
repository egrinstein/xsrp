import numpy as np


def create_uniform_cartesian_grid_2d(dims, n_grid_cells):
    grid = set()
    
    for x in range(n_grid_cells):
        for y in range(n_grid_cells):
            grid.add(GridCell((x, y)))

    return grid


def create_uniform_cartesian_grid_3d(dims, n_grid_cells):
    grid = set()
    
    for x in range(n_grid_cells):
        for y in range(n_grid_cells):
            for z in range(n_grid_cells):
                grid.add(GridCell((x, y, z)))

    return grid


def create_uniform_angular_grid_1d(n_grid_cells):
    pass

def create_uniform_angular_grid_2d(n_grid_cells):
    pass


class GridCell:
    def __init__(self, position, likelihood=None):
        self.position = position
        self.likelihood = likelihood

    def set_likelihood(self, value):
        self.likelihood = value
