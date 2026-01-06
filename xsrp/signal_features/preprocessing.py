import numpy as np
from scipy.signal import butter, filtfilt


def apply_bandpass_filter(
    signals: np.ndarray,
    fs: float,
    lowcut: float = 100.0,
    highcut: float = 7000.0,
    order: int = 4
) -> np.ndarray:
    """Apply bandpass filter to signals.
    
    Parameters
    ----------
    signals : np.ndarray
        Input signals of shape (n_mics, n_samples) or (n_samples,)
    fs : float
        Sampling rate in Hz
    lowcut : float
        High-pass cutoff frequency in Hz (default 100)
    highcut : float
        Low-pass cutoff frequency in Hz (default 7000)
    order : int
        Filter order (default 4)
    
    Returns
    -------
    np.ndarray
        Filtered signals with same shape as input
    """
    nyquist = fs / 2.0
    
    # Validate cutoff frequencies
    if lowcut >= highcut:
        raise ValueError(f"lowcut ({lowcut}) must be less than highcut ({highcut})")
    if highcut > nyquist:
        raise ValueError(f"highcut ({highcut}) must be less than Nyquist frequency ({nyquist})")
    
    # Normalize frequencies
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)  # Cap slightly below Nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter (filtfilt for zero-phase filtering)
    if signals.ndim == 1:
        filtered = filtfilt(b, a, signals)
    else:
        filtered = filtfilt(b, a, signals, axis=1)
    
    return filtered


