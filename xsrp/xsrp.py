import numpy as np

from abc import ABC, abstractmethod


class XSrp(ABC):
    def __init__(self, fs: float, mic_positions=None, room_dims=None, c=343.0):
        self.fs = fs
        self.mic_positions = mic_positions
        self.room_dims = room_dims
        self.c = c

        self.n_mics = len(mic_positions)

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

        signal_features = self.compute_signal_features(mic_signals)

        while True:
            spatial_mapper = self.create_spatial_mapper(mic_positions, candidate_grid)
            srp_map = self.project_features(
                spatial_mapper, signal_features
            )

            estimated_positions, new_candidate_grid = self.grid_search(
                candidate_grid, srp_map, estimated_positions
            )

            if len(new_candidate_grid) == 0:
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
    def create_spatial_mapper(self, mic_positions, candidate_grid):
        pass


    @abstractmethod
    def project_features(self,
                         spatial_mapper,
                         signal_features):
        pass

    @abstractmethod
    def grid_search(self, candidate_grid, srp_map, estimated_positions) -> tuple[np.array, np.array]:
        pass
