# XsrP: eXtensible Steered Response Power

This repository contains the code for the paper:

**A review of the Steered
Response Power
method for sound
source localization**

## The following functionality is currently implemented:

- Conventional SRP-PHAT in the time domain (taking the DFT, applying the phase transform, followed by the IDFT) as described by Dibiase et al. [1]
- Conventional SRP-PHAT in the frequency domain (same as above without the IDFT step) as described by Dibiase et al.
- SRP in the time domain, using temporal cross-correlation without phase transform
- Parabolic interpolation of the cross-correlation function in time
- A simple Volumetric SRP approach which projects the average of N-closest correlation values instead of only the one associated with the microphone pair's Time Difference of Arrival (TDOA)
- Grid creation functions for Positional Source Localization and Direction of Arrival (DOA) Estimation
- Visualization tools
