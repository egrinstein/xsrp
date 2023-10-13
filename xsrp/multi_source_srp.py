import numpy as np


def de_emphasize_peak(cross_correlation: np.ndarray, lags: np.ndarray,
                      peak_location: float, b: float = 2, p: float = 2,
                      return_notch_filter: bool = False):
    """
    De-emphasize a peak in a cross-correlation function by applying an exponential notch filter
    as defined in [1]_.
    
    [1] 1. Brutti, A., Omologo, M. & Svaizer, P.
    Multiple Source Localization Based on Acoustic Map De-Emphasis. J_ASM 2010, 1-17 (2010).


    Parameters
    ----------
    cross_correlation : np.ndarray
        The cross-correlation function.
    lags : np.ndarray
        The lags of the cross-correlation function.
    peak_location : float
        The location of the peak, in samples.
    b : float
        The width of the raised cosine window. Defaults to 2.
    p : float
        The power of the raised cosine window. Defaults to 2.
    return_notch_filter : bool
        Whether to return the notch filter. Defaults to False.
        
    Returns
    -------
    np.ndarray
        The cross-correlation function with the de-emphasized peak.
    """

    window = notch_filter(lags, peak_location, b, p)
    # # Normalize the window so that it sums to 1
    # window /= np.sum(window)

    # Apply the notch filter
    cross_correlation = cross_correlation*window
    
    if return_notch_filter:
        return cross_correlation, window
    else:
        return cross_correlation

def notch_filter(r: float, mu: float, b: float = 2, p: float = 2):
    """
    Create a notch filter as defined in [1]_.
    
    [1] 1. Brutti, A., Omologo, M. & Svaizer, P.
    Multiple Source Localization Based on Acoustic Map De-Emphasis. J_ASM 2010, 1-17 (2010).


    Parameters
    ----------
    r : float
        The index to evaluate the notch filter at.
    mu : float
        The centre of the notch filter.
    b : float
        The width of the raised cosine window.
    p : float
        The power of the raised cosine window.
    
    Returns
    -------
    np.ndarray
        The notch filter.
    """

    return 1 - np.exp(-(np.abs(r - mu)/b)**p)
