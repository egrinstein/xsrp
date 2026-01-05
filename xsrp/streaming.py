import numpy as np

from .xsrp import XSrp
from .tracking import Tracker


class StreamingSrp:
    """Streaming SRP processor for real-time audio processing.
    
    Wraps an XSrp processor (or subclass) to handle frame-based processing
    with integrated tracking.
    
    Parameters
    ----------
    srp_processor : XSrp
        The SRP processor to use for frame processing. Must have a 'grid_type'
        attribute set to 'doa_1D' or 'doa_2D' for DOA estimation.
    tracker : Tracker
        Tracker instance for smoothing DOA estimates (use DummyTracker to disable tracking)
    frame_size : int
        Number of samples per processing window
    hop_size : int, optional
        Number of samples between windows (for overlap). Defaults to frame_size (no overlap).
    """
    
    def __init__(self, srp_processor: XSrp, tracker: Tracker,
                 frame_size: int, hop_size: int = None):
        self.srp_processor = srp_processor
        self.tracker = tracker
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size
        
        # Validate that SRP processor is configured for DOA
        # Check if grid_type attribute exists (for ConventionalSrp and subclasses)
        if hasattr(srp_processor, 'grid_type'):
            if srp_processor.grid_type not in ["doa_1D", "doa_2D"]:
                raise ValueError(
                    "StreamingSrp requires SRP processor with grid_type 'doa_1D' or 'doa_2D'"
                )
    
    def process_frame(self, audio_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process a single audio frame.
        
        Parameters
        ----------
        audio_frame : np.ndarray
            Audio frame of shape (n_mics, frame_size)
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (smoothed_srp_map, estimated_doa)
            - smoothed_srp_map: Smoothed SRP map values for the grid
            - estimated_doa: DOA estimate from the smoothed map (cartesian unit vector or azimuth)
        """
        # Process frame with SRP
        _, srp_map, _ = self.srp_processor.forward(audio_frame)
        
        # Apply smoothing to the SRP map (not the position)
        smoothed_srp_map = self.tracker.update(srp_map)
        
        # Find the peak in the smoothed map for display
        max_idx = np.argmax(smoothed_srp_map)
        grid_positions = self.srp_processor.candidate_grid.asarray()
        estimated_doa = grid_positions[max_idx]
        
        return smoothed_srp_map, estimated_doa
    
    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()

    def __call__(self, mic_signals, mic_positions=None, room_dims=None) -> tuple[np.ndarray, np.ndarray]:
        return self.process_frame(mic_signals)