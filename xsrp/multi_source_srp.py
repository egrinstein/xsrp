import numpy as np


from .xsrp import XSrp


class MultiSourceSrp(XSrp):
    def __init__(self, fs: float,
        grid_type, n_grid_cells,
        mic_positions=None, room_dims=None, c=343,
        mode="gcc_phat_time",
        interpolation=False,
        n_average_samples=1,
        n_dft_bins=1024,
        freq_cutoff_in_hz=None):

        available_modes = ["gcc_phat_time", "cross_correlation"]
        if mode not in available_modes:
            raise ValueError(
                "mode must be one of {}".format(available_modes)
            )