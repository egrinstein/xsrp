from abc import ABC, abstractmethod


class XSrp(ABC):
    def __init__(self, fs: float, mic_positions=None, room_dims=None, c=343.0):
        self.fs = fs
        self.mic_positions = mic_positions
        self.room_dims = room_dims
        self.c = c

        self.n_mics = len(mic_positions)

        self.candidate_grid = self.create_initial_candidate_grid(room_dims)

    def forward(self, mic_signals, mic_positions=None, room_dims=None):
        if mic_positions is None:
            mic_positions = self.mic_positions
        if room_dims is None:
            room_dims = self.room_dims
        
        if mic_positions is None or room_dims is None:
            raise ValueError(
                """
                mic_positions and room_dims must be specified
                either in the constructor or in the forward method
                """
            )
        
        candidate_grid = self.candidate_grid.copy()

        estimated_positions = set()

        time_space_mapper = self.create_time_space_mapper(mic_positions, candidate_grid)

        temporal_features = self.compute_temporal_features(mic_signals)

        while len(candidate_grid) > 0:
            candidate_grid = self.project_temporal_features_in_space(
                candidate_grid, time_space_mapper, temporal_features, self.fs
            )

            estimated_positions, candidate_grid = self.grid_search(
                candidate_grid, estimated_positions
            )

        return estimated_positions, candidate_grid
    
    @abstractmethod
    def create_initial_candidate_grid(self, room_dims):
        pass

    @abstractmethod
    def create_time_space_mapper(self, mic_positions, candidate_grid):
        pass

    @abstractmethod
    def compute_temporal_features(self, mic_signals):
        pass

    @abstractmethod
    def project_temporal_features_in_space(self,
                                           candidate_grid,
                                           time_space_mapper,
                                           temporal_features,
                                           fs):
        pass
