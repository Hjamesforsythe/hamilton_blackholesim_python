# Black Hole Simulation

Python implementation of black hole physics using Schwarzschild geodesic equations. Includes both 2D ray tracing and 3D OpenGL visualization.

## Overview

This project simulates light ray paths around black holes using general relativity. The 2D version traces individual light rays, while the 3D version uses GPU compute shaders for visualization.

## Features

- Schwarzschild metric geodesic integration
- RK4 numerical integration with adaptive stepping
- 2D ray tracing with gravitational lensing
- 3D OpenGL visualization with orbital camera
- GPU compute shaders for ray tracing

## Installation

1. Clone the repository
2. Create conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate blackholesim
   ```

## Usage

Run 2D simulation:
```bash
python blackhole_2d.py
```

Run 3D simulation:
```bash
python blackhole_3d.py
```

## Controls

**3D Simulation:**
- Left mouse drag: Orbit camera
- Mouse scroll: Zoom
- ESC: Exit

**2D Simulation:**
- ESC: Exit

## Files

- `blackhole_2d.py` - 2D ray tracing simulation
- `blackhole_3d.py` - 3D OpenGL visualization  
- `vector.py` - Vector math utilities
- `utilities.py` - General utilities
- `shaders/geodesic.comp` - GPU ray tracing compute shader
- `shaders/grid.vert`, `shaders/grid.frag` - Grid rendering shaders

## Requirements

- Python 3.11+
- NumPy, SciPy, PyOpenGL, GLFW
- OpenGL 4.3+ (for 3D simulation compute shaders)

## Resources
Largely inspired by https://github.com/kavan010/black_hole
With resources from https://www.youtube.com/@GetIntoGameDev
