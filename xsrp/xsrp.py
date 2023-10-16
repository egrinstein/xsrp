import numpy as np

from abc import ABC, abstractmethod

from xsrp.grids import Grid


class XSrp(ABC):
    def __init__(self, fs: float, mic_positions=None, room_dims=None, c=343.0):
        self.fs = fs
        self.mic_positions = mic_positions
        self.room_dims = room_dims
        self.c = c

        self.n_mics = len(mic_positions)

        # 0. Create the initial grid of candidate positions
        self.candidate_grid = self.create_initial_candidate_grid(room_dims)

    def forward(self, mic_signals, mic_positions=None, room_dims=None) -> tuple[set, np.array]:
        if mic_positions is None:
            mic_positions = self.mic_positions
        if room_dims is None:
            room_dims = self.room_dims
        
        if mic_positions is None:
            raise ValueError(
                """
                mic_positions and room_dims must be specified
                either in the constructor or in the forward method
                """
            )
        
        candidate_grid = self.candidate_grid

        estimated_positions = np.array([])

        # 1. Compute the signal features (usually, GCC-PHAT)
        signal_features = self.compute_signal_features(mic_signals)

        while True:
            # 2. Project the signal features into space, i.e., create the SRP map
            srp_map = self.create_srp_map(
                mic_positions, candidate_grid, signal_features
            )

            # 3. Find the source position in the SRP map
            estimated_positions, new_candidate_grid, signal_features = self.grid_search(
                candidate_grid, srp_map, estimated_positions, signal_features
            )

            # 4. Update the candidate grid
            if len(new_candidate_grid) == 0:
                # If the new candidate grid is empty, we have found all the sources
                break
            else:
                candidate_grid = new_candidate_grid

        return estimated_positions, srp_map, candidate_grid
    
    @abstractmethod
    def compute_signal_features(self, mic_signals):
        pass

    @abstractmethod
    def create_initial_candidate_grid(self, room_dims):
        pass

    @abstractmethod
    def create_srp_map(self,
                       mic_positions: np.array,
                       candidate_grid: Grid,
                       signal_features: np.array):
        pass

    @abstractmethod
    def grid_search(self, candidate_grid, srp_map,
                    estimated_positions, signal_features) -> tuple[np.array, np.array]:
        pass
