import numpy as np


def compute_spatial_coherence(mic_signals_dft: np.ndarray) -> np.ndarray:
    """Compute magnitude squared coherence (MSC) across all microphone pairs for each frequency.
    
    Parameters
    ----------
    mic_signals_dft : np.ndarray (n_mics, n_frequencies)
        Frequency-domain representation of microphone signals (complex-valued)
    
    Returns
    -------
    coherence : np.ndarray (n_frequencies,)
        Average magnitude squared coherence across all mic pairs for each frequency
    """
    n_mics, n_freqs = mic_signals_dft.shape
    coherences = []
    
    for f in range(n_freqs):
        freq_coherences = []
        
        for i in range(n_mics):
            for j in range(i + 1, n_mics):
                # Cross-spectral density
                Gxy = mic_signals_dft[i, f] * np.conj(mic_signals_dft[j, f])
                # Auto-spectral densities
                Gxx = np.abs(mic_signals_dft[i, f])**2
                Gyy = np.abs(mic_signals_dft[j, f])**2
                # MSC (Magnitude Squared Coherence)
                msc = np.abs(Gxy)**2 / (Gxx * Gyy + 1e-10)
                freq_coherences.append(msc)
        
        # Average across all mic pairs for this frequency
        coherences.append(np.mean(freq_coherences))
    
    return np.array(coherences)


def compute_sparsity(srp_map: np.ndarray) -> float:
    """Compute Gini coefficient as a sparsity metric.
    
    Higher Gini coefficient indicates more sparse/peaked distribution.
    
    Parameters
    ----------
    srp_map : np.ndarray (n_cells,)
        SRP map values for a single frequency
    
    Returns
    -------
    sparsity : float
        Gini coefficient (0 = uniform, 1 = maximally sparse)
    """
    # Sort values in ascending order
    values = np.sort(srp_map)
    n = len(values)
    
    if n == 0 or np.sum(values) == 0:
        return 0.0
    
    # Compute Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
    
    return gini


def compute_par(srp_map: np.ndarray) -> float:
    """Compute peak-to-average ratio (PAR).
    
    Higher PAR indicates a more dominant peak relative to the average.
    
    Parameters
    ----------
    srp_map : np.ndarray (n_cells,)
        SRP map values for a single frequency
    
    Returns
    -------
    par : float
        Peak-to-average ratio (max / mean)
    """
    if len(srp_map) == 0:
        return 0.0
    
    mean_val = np.mean(srp_map)
    if mean_val == 0:
        return 0.0
    
    return np.max(srp_map) / mean_val


def compute_frequency_weights(
    per_freq_srp_maps: np.ndarray,
    mic_signals_dft: np.ndarray = None,
    method: str = 'coherence'
) -> np.ndarray:
    """Compute frequency weights using the specified method.
    
    Parameters
    ----------
    per_freq_srp_maps : np.ndarray (n_cells, n_frequencies)
        SRP maps for each frequency (before summation across frequencies)
    mic_signals_dft : np.ndarray (n_mics, n_frequencies), optional
        Frequency-domain microphone signals (required for 'coherence' method)
    method : str, optional
        Weighting method: 'coherence', 'sparsity', 'par', or None. Defaults to 'coherence'.
    
    Returns
    -------
    weights : np.ndarray (n_frequencies,)
        Weights for each frequency bin
    """
    n_cells, n_freqs = per_freq_srp_maps.shape
    
    if method is None or method == 'none':
        # Uniform weighting (no weighting)
        return np.ones(n_freqs)
    
    weights = []
    
    if method == 'coherence':
        if mic_signals_dft is None:
            raise ValueError("mic_signals_dft is required for coherence weighting")
        weights = compute_spatial_coherence(mic_signals_dft)
    
    elif method == 'sparsity':
        for f in range(n_freqs):
            srp_f = per_freq_srp_maps[:, f]
            # Shift to non-negative by subtracting minimum, since sparsity needs non-negative values
            # This preserves the relative structure while ensuring non-negative
            srp_f_shifted = srp_f - np.min(srp_f)
            if np.sum(srp_f_shifted) > 0:
                sparsity = compute_sparsity(srp_f_shifted)
            else:
                sparsity = 0.0
            weights.append(sparsity)
        weights = np.array(weights)
    
    elif method == 'par':
        for f in range(n_freqs):
            srp_f = per_freq_srp_maps[:, f]
            # Shift to non-negative for PAR calculation
            srp_f_shifted = srp_f - np.min(srp_f)
            par = compute_par(srp_f_shifted)
            weights.append(par)
        weights = np.array(weights)
    
    else:
        raise ValueError(f"Unknown weighting method: {method}. Must be 'coherence', 'sparsity', 'par', or None")
    
    # Normalize weights to prevent bias (optional, but helps with stability)
    if np.sum(weights) > 0:
        weights = weights / (np.sum(weights) / n_freqs)  # Normalize to mean of 1.0
    
    return weights

