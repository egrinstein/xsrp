import numpy as np
import pyaudio
import h5py
from pathlib import Path
from typing import Optional

from .conventional_srp import ConventionalSrp
from .signal_features.preprocessing import apply_bandpass_filter


def calculate_aliasing_limit(mic_positions: np.ndarray, c: float = 343.0, margin: float = 0.9) -> Optional[int]:
    """Calculate the spatial aliasing frequency limit for a microphone array.
    
    Aliasing occurs when the wavelength is smaller than 2 * minimum_distance.
    f < c / (2 * d_min)
    
    Parameters
    ----------
    mic_positions : np.ndarray
        Microphone positions of shape (n_mics, 3) or (n_mics, 2)
    c : float
        Speed of sound in m/s (default 343.0)
    margin : float
        Safety margin factor to apply to the limit (default 0.9)
        
    Returns
    -------
    int or None
        Suggested frequency limit in Hz, or None if not enough microphones
    """
    if len(mic_positions) < 2:
        return None
        
    # Calculate minimum pairwise distance
    dists = []
    for i in range(len(mic_positions)):
        for j in range(i + 1, len(mic_positions)):
            dist = np.linalg.norm(mic_positions[i] - mic_positions[j])
            if dist > 0:
                dists.append(dist)
    
    if not dists:
        return None
        
    min_dist = min(dists)
    
    # Aliasing limit: f < c / (2 * d)
    aliasing_freq = c / (2 * min_dist)
    
    # Apply safety margin
    suggested_limit = int(aliasing_freq * margin)
    
    return suggested_limit


def compute_noise_floor(
    mic_positions: Optional[np.ndarray] = None,
    fs: int = 16000,
    frame_size: int = 1024,
    duration_seconds: float = 5.0,
    n_azimuth_cells: int = 72,
    n_average_samples: int = 5,
    sharpening: float = 1.0,
    mode: str = "gcc_phat_time",
    interpolation: bool = True,
    device_index: Optional[int] = None,
    ignore_channels: Optional[list] = None,
    progress_callback: Optional[callable] = None,
    filter_enabled: bool = True,
    filter_lowcut: float = 100.0,
    filter_highcut: float = 7000.0,
    srp_processor: Optional[ConventionalSrp] = None,
    frequency_weighting: Optional[str] = None
) -> np.ndarray:
    """Compute noise floor by averaging SRP maps from silent environment.
    
    Parameters
    ----------
    mic_positions : np.ndarray, optional
        Microphone positions of shape (n_mics, n_dimensions). Required if srp_processor is None.
    fs : int
        Sampling rate in Hz
    frame_size : int
        Number of samples per frame
    duration_seconds : float
        Duration of recording in seconds
    n_azimuth_cells : int
        Number of azimuth cells for DOA grid
    n_average_samples : int
        Number of cross-correlation samples to average over
    sharpening : float
        Sharpening exponent for SRP map
    mode : str
        SRP mode ('gcc_phat_time', 'gcc_phat_freq', 'cross_correlation')
    interpolation : bool
        Whether to use fractional sample interpolation
    device_index : int, optional
        Audio device index. If None, uses default input device.
    ignore_channels : list, optional
        List of channel indices to ignore
    progress_callback : callable, optional
        Callback function(current_frame, total_frames) for progress updates
    filter_enabled : bool, optional
        Whether to apply bandpass filtering (default True)
    filter_lowcut : float, optional
        High-pass cutoff frequency in Hz (default 100.0)
    filter_highcut : float, optional
        Low-pass cutoff frequency in Hz (default 7000.0)
    srp_processor : ConventionalSrp, optional
        Existing SRP processor to use. If provided, other SRP parameters are ignored.
    frequency_weighting : str, optional
        Frequency weighting method for 'gcc_phat_freq' mode (e.g., 'coherence').
    
    Returns
    -------
    np.ndarray
        Averaged noise floor SRP map of shape (n_azimuth_cells,)
    """
    
    if srp_processor is None:
        if mic_positions is None:
            raise ValueError("mic_positions must be provided if srp_processor is None")
            
        # For 1D DOA grid, use 2D mic positions (x, y only)
        mic_positions_2d = mic_positions[:, :2] if mic_positions.shape[1] > 2 else mic_positions
        
        # Create SRP processor
        srp_processor = ConventionalSrp(
            fs=fs,
            grid_type="doa_1D",
            n_grid_cells=n_azimuth_cells,
            mic_positions=mic_positions_2d,
            mode=mode,
            interpolation=interpolation,
            n_average_samples=n_average_samples,
            sharpening=sharpening,
            frequency_weighting=frequency_weighting
        )
    
    # Initialize audio
    audio = pyaudio.PyAudio()
    
    try:
        if device_index is None:
            # Find default input device
            device_index = audio.get_default_input_device_info()['index']
        
        device_info = audio.get_device_info_by_index(device_index)
        device_channels = device_info['maxInputChannels']
        
        # Open audio stream
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=device_channels,
            rate=fs,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=frame_size,
            stream_callback=None
        )
        
        stream.start_stream()
        
        # Calculate number of frames
        n_frames = int(duration_seconds * fs / frame_size)
        
        # Accumulate SRP maps
        srp_maps = []
        
        print(f"Recording noise floor for {duration_seconds} seconds ({n_frames} frames)...")
        
        for frame_idx in range(n_frames):
            # Read audio data
            audio_data = stream.read(frame_size, exception_on_overflow=False)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            # Reshape to (channels, samples)
            audio_array = audio_array.reshape(frame_size, device_channels).T
            
            # Filter out ignored channels
            if ignore_channels is not None:
                valid_channels = [i for i in range(device_channels) if i not in ignore_channels]
                audio_frame = audio_array[valid_channels, :]
            else:
                audio_frame = audio_array
            
            # Apply bandpass filter if enabled
            if filter_enabled:
                try:
                    audio_frame = apply_bandpass_filter(
                        audio_frame,
                        fs,
                        lowcut=filter_lowcut,
                        highcut=filter_highcut
                    )
                except Exception as e:
                    print(f"Filter error during calibration: {e}")
            
            # Process frame to get SRP map
            _, srp_map, _ = srp_processor.forward(audio_frame)
            srp_maps.append(srp_map)
            
            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(frame_idx + 1, n_frames)
        
        stream.stop_stream()
        stream.close()
        
        # Average all SRP maps
        noise_floor = np.mean(srp_maps, axis=0)
        
        print(f"Computed noise floor from {len(srp_maps)} frames")
        
        return noise_floor
        
    finally:
        audio.terminate()


def save_noise_floor(noise_floor: np.ndarray, file_path: str, metadata: Optional[dict] = None):
    """Save noise floor SRP map to HDF5 file.
    
    Parameters
    ----------
    noise_floor : np.ndarray
        Noise floor SRP map
    file_path : str
        Path to save the HDF5 file
    metadata : dict, optional
        Additional metadata to save (e.g., fs, n_azimuth_cells, etc.)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        # Save noise floor
        f.create_dataset('noise_floor', data=noise_floor)
        
        # Save metadata if provided
        if metadata is not None:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
                elif isinstance(value, np.ndarray):
                    f.create_dataset(f'metadata/{key}', data=value)
                else:
                    # Try to convert to string
                    f.attrs[key] = str(value)


def load_noise_floor(file_path: str) -> tuple[np.ndarray, dict]:
    """Load noise floor SRP map from HDF5 file.
    
    Parameters
    ----------
    file_path : str
        Path to the HDF5 file
    
    Returns
    -------
    tuple[np.ndarray, dict]
        (noise_floor, metadata) where metadata is a dictionary of saved attributes
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Noise floor file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        noise_floor = f['noise_floor'][:]
        
        # Load metadata
        metadata = {}
        for key in f.attrs.keys():
            metadata[key] = f.attrs[key]
        
        # Load metadata datasets if they exist
        if 'metadata' in f:
            for key in f['metadata'].keys():
                metadata[key] = f['metadata'][key][:]
    
    return noise_floor, metadata
