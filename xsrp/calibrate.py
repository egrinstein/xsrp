import numpy as np
from pathlib import Path
from typing import Optional

# Optional imports for GUI/calibration features
try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import h5py
except ImportError:
    h5py = None

from .xsrp import XSrp
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


def record_ambient_noise(
    fs: int = 16000,
    duration_seconds: float = 5.0,
    device_index: Optional[int] = None,
    chunk_size: int = 1024,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Record raw audio from the environment for calibration.
    
    Parameters
    ----------
    fs : int
        Sampling rate in Hz
    duration_seconds : float
        Duration of recording in seconds
    device_index : int, optional
        Audio device index. If None, uses default input device.
    chunk_size : int
        Size of chunks to read from stream
    progress_callback : callable, optional
        Callback function(current_chunk, total_chunks)
    
    Returns
    -------
    np.ndarray
        Raw audio data of shape (n_samples, n_channels)
    
    Raises
    ------
    ImportError
        If pyaudio is not installed. Install with: pip install xsrp[gui]
    """
    if pyaudio is None:
        raise ImportError(
            "pyaudio is required for recording ambient noise. "
            "Install it with: pip install xsrp[gui]"
        )
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
            frames_per_buffer=chunk_size,
            stream_callback=None
        )
        
        stream.start_stream()
        
        # Calculate number of chunks
        n_chunks = int(duration_seconds * fs / chunk_size) + 1
        
        frames = []
        
        print(f"Recording ambient noise for {duration_seconds} seconds...")
        
        for i in range(n_chunks):
            try:
                data = stream.read(chunk_size, exception_on_overflow=False)
                # Convert to float32 normalized
                chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                frames.append(chunk)
                
                if progress_callback:
                    progress_callback(i + 1, n_chunks)
                    
            except Exception as e:
                print(f"Error reading audio chunk: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        
        if not frames:
            raise RuntimeError("No audio recorded")
            
        # Concatenate all chunks
        raw_data = np.concatenate(frames)
        
        # Reshape to (n_samples, n_channels)
        # stream.read gives interleaved data [c1s1, c2s1, c1s2, c2s2, ...]
        n_samples = len(raw_data) // device_channels
        audio_data = raw_data[:n_samples * device_channels].reshape(n_samples, device_channels)
        
        return audio_data
        
    finally:
        audio.terminate()


def compute_noise_floor_map(
    audio_data: np.ndarray,
    srp_processor: XSrp,
    frame_size: int = 1024,
    hop_size: int = 512,
    fs: float = 16000,
    filter_enabled: bool = True,
    filter_lowcut: float = 100.0,
    filter_highcut: float = 7000.0,
    ignore_channels: Optional[list] = None,
    progress_callback: Optional[callable] = None
) -> np.ndarray:
    """Compute average SRP map from recorded ambient noise audio.
    
    Parameters
    ----------
    audio_data : np.ndarray
        Raw audio data of shape (n_samples, n_channels)
    srp_processor : XSrp
        Configured SRP processor to use
    frame_size : int
        Processing frame size
    hop_size : int
        Processing hop size
    fs : float
        Sampling rate
    filter_enabled : bool
        Whether to apply bandpass filtering
    filter_lowcut : float
        High-pass cutoff
    filter_highcut : float
        Low-pass cutoff
    ignore_channels : list, optional
        List of channel indices to ignore
    progress_callback : callable, optional
        Callback function(current_frame, total_frames). 
        Should return True to continue, False to cancel.
    
    Returns
    -------
    np.ndarray
        Averaged SRP map
    """
    n_samples, n_channels = audio_data.shape
    
    # Calculate number of frames
    n_frames = (n_samples - frame_size) // hop_size + 1
    
    if n_frames <= 0:
        raise ValueError("Audio data too short for given frame size")
    
    srp_maps = []
    
    # Process frames
    for i in range(n_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        
        # Extract frame: (frame_size, n_channels)
        frame = audio_data[start_idx:end_idx, :]
        
        # Transpose to (n_channels, frame_size) for processing
        frame_T = frame.T
        
        # Filter out ignored channels
        if ignore_channels is not None:
            valid_channels = [ch for ch in range(n_channels) if ch not in ignore_channels]
            frame_proc = frame_T[valid_channels, :]
        else:
            frame_proc = frame_T
            
        # Apply bandpass filter
        if filter_enabled:
            try:
                frame_proc = apply_bandpass_filter(
                    frame_proc,
                    fs,
                    lowcut=filter_lowcut,
                    highcut=filter_highcut
                )
            except Exception:
                pass  # Skip filter errors
        
        # Compute SRP map
        _, srp_map, _ = srp_processor.forward(frame_proc)
        srp_maps.append(srp_map)
        
        if progress_callback:
            # Callback can return False to cancel
            result = progress_callback(i + 1, n_frames)
            if result is False:
                # User cancelled, return partial result or raise
                break
    
    # Average maps
    if not srp_maps:
        return np.zeros(srp_processor.n_grid_cells)
        
    return np.mean(srp_maps, axis=0)


def save_ambient_noise(audio_data: np.ndarray, file_path: str, fs: int):
    """Save ambient noise audio to HDF5 file.
    
    Parameters
    ----------
    audio_data : np.ndarray
        Raw audio data (n_samples, n_channels)
    file_path : str
        Path to save
    fs : int
        Sampling rate
    
    Raises
    ------
    ImportError
        If h5py is not installed. Install with: pip install xsrp[gui]
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for saving ambient noise. "
            "Install it with: pip install xsrp[gui]"
        )
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('ambient_audio', data=audio_data)
        f.attrs['fs'] = fs


def load_ambient_noise(file_path: str) -> tuple[Optional[np.ndarray], int]:
    """Load ambient noise audio from HDF5 file.
    
    Parameters
    ----------
    file_path : str
        Path to HDF5 file
        
    Returns
    -------
    tuple[np.ndarray, int]
        (audio_data, fs). audio_data may be None if file is old format.
    
    Raises
    ------
    ImportError
        If h5py is not installed. Install with: pip install xsrp[gui]
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for loading ambient noise. "
            "Install it with: pip install xsrp[gui]"
        )
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with h5py.File(file_path, 'r') as f:
        if 'ambient_audio' in f:
            return f['ambient_audio'][:], f.attrs.get('fs', 16000)
        else:
            # Old format or invalid file
            return None, 0
