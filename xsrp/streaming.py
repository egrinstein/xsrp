import numpy as np

from .xsrp import XSrp
from .tracking import Tracker
from .signal_features.preprocessing import apply_bandpass_filter


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
    fs : float, optional
        Sampling rate in Hz (required for filtering)
    filter_enabled : bool, optional
        Whether to apply bandpass filtering (default False)
    filter_lowcut : float, optional
        High-pass cutoff frequency in Hz (default 100.0)
    filter_highcut : float, optional
        Low-pass cutoff frequency in Hz (default 7000.0)
    """
    
    def __init__(self, srp_processor: XSrp, tracker: Tracker,
                 frame_size: int, hop_size: int = None, window_func=None,
                 fs: float = 16000, filter_enabled: bool = False,
                 filter_lowcut: float = 100.0, filter_highcut: float = 7000.0):
        self.srp_processor = srp_processor
        self.tracker = tracker
        self.frame_size = frame_size
        self.hop_size = hop_size if hop_size is not None else frame_size
        
        # Filtering parameters
        self.fs = fs
        self.filter_enabled = filter_enabled
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        
        # Pre-compute window if a function is provided
        if window_func is not None:
            self.window = window_func(frame_size)
        else:
            self.window = None
        
        # Validate that SRP processor is configured for DOA
        # Check if grid_type attribute exists (for XSrp and subclasses)
        if hasattr(srp_processor, 'grid_type'):
            if srp_processor.grid_type not in ["doa_1D", "doa_2D"]:
                raise ValueError(
                    "StreamingSrp requires SRP processor with grid_type 'doa_1D' or 'doa_2D'"
                )
                
        # Internal buffer for streaming
        self.buffer = None

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process an arbitrary chunk of audio data, buffering as needed.
        
        Parameters
        ----------
        audio_chunk : np.ndarray
            Audio chunk of shape (n_mics, n_samples)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (smoothed_srp_map, estimated_doa) or (None, None) if not enough data
        """
        # Apply filtering to the incoming chunk if enabled
        # Note: applying stateless filter (filtfilt) to chunks can cause boundary artifacts.
        # Ideally, a stateful filter (sosfilt) should be used for streaming.
        # However, for demo purposes and zero-phase requirement, we use filtfilt here
        # matching the original implementation behavior.
        if self.filter_enabled:
            try:
                audio_chunk = apply_bandpass_filter(
                    audio_chunk,
                    self.fs,
                    lowcut=self.filter_lowcut,
                    highcut=self.filter_highcut
                )
            except Exception as e:
                # Fallback or log if filter fails (e.g. invalid design)
                print(f"Filter error: {e}")

        n_mics, n_samples = audio_chunk.shape
        
        # Initialize buffer if needed
        if self.buffer is None:
            self.buffer = np.zeros((n_mics, self.frame_size), dtype=audio_chunk.dtype)
            
        # Roll buffer and add new samples
        # Note: This assumes n_samples <= frame_size. If chunk is larger, we should process multiple times,
        # but for typical real-time use, chunk <= frame_size.
        if n_samples >= self.frame_size:
            # If chunk is larger than frame, take the last frame_size samples
            self.buffer = audio_chunk[:, -self.frame_size:]
        else:
            self.buffer = np.roll(self.buffer, -n_samples, axis=1)
            self.buffer[:, -n_samples:] = audio_chunk
            
        # Process the current full buffer
        return self.process_frame(self.buffer)
    
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
        # Apply windowing if configured
        if self.window is not None:
            # Broadcast window to all channels: (n_mics, frame_size) * (frame_size,)
            audio_frame = audio_frame * self.window

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
