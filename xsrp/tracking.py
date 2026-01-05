import numpy as np
from abc import ABC, abstractmethod


class Tracker(ABC):
    """Abstract base class for tracking DOA estimates over time."""
    
    @abstractmethod
    def update(self, observation: np.ndarray) -> np.ndarray:
        """Update tracker with new observation.
        
        Parameters
        ----------
        observation : np.ndarray
            New DOA observation (e.g., azimuth angle in radians or cartesian unit vector)
        
        Returns
        -------
        np.ndarray
            Smoothed/tracked estimate
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state."""
        pass


class DummyTracker(Tracker):
    """Dummy tracker that passes through observations without modification.
    
    Useful for disabling tracking functionality.
    """
    
    def update(self, observation: np.ndarray) -> np.ndarray:
        """Return observation unchanged.
        
        Parameters
        ----------
        observation : np.ndarray
            DOA observation
        
        Returns
        -------
        np.ndarray
            Same as input observation
        """
        return observation.copy()
    
    def reset(self):
        """No state to reset."""
        pass


class ExponentialSmoothingTracker(Tracker):
    """Exponential smoothing tracker for SRP maps or DOA estimates.
    
    Applies exponential smoothing: s_t = alpha * x_t + (1-alpha) * s_{t-1}
    where s_t is the smoothed value, x_t is the observation, and alpha is the smoothing factor.
    
    Can be used to smooth SRP maps (arrays) or DOA estimates (vectors).
    
    Parameters
    ----------
    alpha : float, optional
        Smoothing factor between 0 and 1. Higher values give more weight to recent observations.
        Defaults to 0.7.
    """
    
    def __init__(self, alpha: float = 0.7):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.smoothed_value = None
    
    def update(self, observation: np.ndarray) -> np.ndarray:
        """Update tracker with new observation.
        
        Parameters
        ----------
        observation : np.ndarray
            New observation (can be SRP map array or DOA estimate vector)
        
        Returns
        -------
        np.ndarray
            Smoothed estimate (same shape as observation)
        """
        if self.smoothed_value is None:
            # First observation: initialize with observation
            self.smoothed_value = observation.copy()
        else:
            # Ensure shapes match
            if self.smoothed_value.shape != observation.shape:
                # Reset if shape changed (e.g., grid changed)
                self.smoothed_value = observation.copy()
            else:
                # Apply exponential smoothing element-wise
                self.smoothed_value = (
                    self.alpha * observation + (1 - self.alpha) * self.smoothed_value
                )
        
        return self.smoothed_value.copy()
    
    def reset(self):
        """Reset tracker state."""
        self.smoothed_value = None

