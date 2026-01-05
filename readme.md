# XSRP: eXtensible Steered Response Power

This repository contains the code for the paper:

[Steered Response Power for Sound Source Localization: A Tutorial Review
](https://arxiv.org/abs/2405.02991)

## The following functionality is currently implemented:

- Conventional SRP-PHAT in the time domain (taking the DFT, applying the phase transform, followed by the IDFT) as described by Dibiase et al. [1]
- Conventional SRP-PHAT in the frequency domain (same as above without the IDFT step) as described by Dibiase et al.
- SRP in the time domain, using temporal cross-correlation without phase transform
- Parabolic interpolation of the cross-correlation function in time
- A simple Volumetric SRP approach which projects the average of N-closest correlation values instead of only the one associated with the microphone pair's Time Difference of Arrival (TDOA)
- Grid creation functions for Positional Source Localization and Direction of Arrival (DOA) Estimation
- Visualization tools

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for dependency management. Change directory to the ```xsrp``` folder and run the following commands:

1. Install uv (if not already installed): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install dependencies: `uv sync`
3. Activate the virtual environment: `source .venv/bin/activate` (or use `uv run` to run commands in the environment)

You can then optionally run the tests to verify that everything is working correctly:

4. `uv run pytest tests` (or `make test`)

You may want to check the images that were generated in the ```tests/temp``` folder.

