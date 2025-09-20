import math
import logging
from typing import Dict
from vector import Vec3
import OpenGL.GL as gl
import OpenGL
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import scipy as sp
from scipy import constants as phys

"""
Comprehensive utility functions for the black hole simulation.
Organized into logical sections for different functionality areas.
"""

# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max bounds."""
        return max(min_val, min(value, max_val))

# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

class MatrixUtils:
    """Matrix operations for 3D graphics."""
    
    @staticmethod
    def create_perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    @staticmethod
    def create_look_at_matrix(eye: Vec3, target: Vec3, up: Vec3) -> np.ndarray:
        """Create look-at view matrix."""
        f = Vec3(target.x - eye.x, target.y - eye.y, target.z - eye.z).normalized()
        s = f.cross(up).normalized()
        u = s.cross(f)
        
        return np.array([
            [s.x, u.x, -f.x, 0],
            [s.y, u.y, -f.y, 0],
            [s.z, u.z, -f.z, 0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1]
        ], dtype=np.float32)

# =============================================================================
# OPENGL UTILITIES
# =============================================================================

class ShaderUtils:
    """Utilities for OpenGL shader management."""
    
    @staticmethod
    def create_shader_program(vertex_filepath: str, fragment_filepath: str) -> int:
        """Create shader program from vertex and fragment shader files."""
        vertex_module = ShaderUtils.create_shader_module(vertex_filepath, gl.GL_VERTEX_SHADER)
        fragment_module = ShaderUtils.create_shader_module(fragment_filepath, gl.GL_FRAGMENT_SHADER)

        shader = compileProgram(vertex_module, fragment_module)
        gl.glDeleteShader(vertex_module)
        gl.glDeleteShader(fragment_module)

        return shader
    
    @staticmethod
    def create_shader_module(filepath: str, module_type: int) -> int:
        """Create individual shader module from file."""
        with open(filepath, "r") as f:
            source_code = f.readlines()
        return compileShader(source_code, module_type)
    
    @staticmethod
    def create_compute_program(shader_path: str) -> int:
        """Load and compile compute shader program."""
        try:
            with open(shader_path, 'r') as f:
                compute_source = f.readlines()
        except FileNotFoundError:
            raise RuntimeError(f"Compute shader file not found: {shader_path}")
        
        compute_shader = compileShader(compute_source, gl.GL_COMPUTE_SHADER)
        program = compileProgram(compute_shader)
        
        return program
    
    @staticmethod
    def create_shader_program_from_source(vertex_source: str, fragment_source: str) -> int:
        """Create shader program from source strings."""
        vertex_shader = compileShader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_source, gl.GL_FRAGMENT_SHADER)
        
        program = compileProgram(vertex_shader, fragment_shader)
        
        return program

class BufferUtils:
    """Utilities for OpenGL buffer management."""
    
    @staticmethod
    def create_vertex_data_type():
        """Create numpy dtype for vertex data."""
        return np.dtype({
            "names": ["x", "y", "z", "color"],
            "formats": [np.float32, np.float32, np.float32, np.uint32],
            "offsets": [0, 4, 8, 12],
            "itemsize": 16
        })

# =============================================================================
# PHYSICS UTILITIES
# =============================================================================

class PhysicsConstants:
    """Physical constants for black hole calculations."""
    
    # Fundamental constants
    C = phys.c  # Speed of light (m/s)
    G = phys.G  # Gravitational constant (m³/kg⋅s²)
    PI = math.pi
    
    # Sagittarius A* properties
    SAGITTARIUS_A_MASS = 8.54e36  # kg (4.31M solar masses)
    
    @classmethod
    def schwarzschild_radius(cls, mass: float) -> float:
        """Calculate Schwarzschild radius: r_s = 2GM/c²"""
        return 2.0 * cls.G * mass / (cls.C ** 2)

class BlackHoleUtils:
    """Utility functions for black hole physics calculations."""
    
    @staticmethod
    def is_inside_event_horizon(position: Vec3, black_hole_mass: float) -> bool:
        """Check if a position is inside the event horizon."""
        r = position.length()
        r_s = PhysicsConstants.schwarzschild_radius(black_hole_mass)
        return r <= r_s * 1.01  # Small buffer for numerical precision
    
    @staticmethod
    def gravitational_redshift(radius: float, schwarzschild_radius: float) -> float:
        """Calculate gravitational redshift factor at given radius."""
        if radius <= schwarzschild_radius:
            return 0.0  # Infinite redshift at event horizon
        return math.sqrt(1.0 - schwarzschild_radius / radius)
    
    @staticmethod
    def escape_velocity(radius: float, mass: float) -> float:
        """Calculate escape velocity at given radius from mass."""
        return math.sqrt(2.0 * PhysicsConstants.G * mass / radius)

class GeodesicUtils:
    """Utilities for geodesic calculations."""
    
    @staticmethod
    def conserved_energy(r: float, dr: float, r_s: float) -> float:
        """Calculate conserved energy for geodesic."""
        f = 1.0 - r_s / r
        # Simplified for radial motion
        return f * math.sqrt(dr * dr / (f * f))
    
    @staticmethod
    def conserved_angular_momentum(r: float, theta: float, dphi: float) -> float:
        """Calculate conserved angular momentum for geodesic."""
        return r * r * math.sin(theta) * dphi
    
    @staticmethod
    def effective_potential(r: float, L: float, r_s: float) -> float:
        """Calculate effective potential for geodesic motion."""
        return (1.0 - r_s / r) * (1.0 + L * L / (r * r))

# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

class SystemUtils:
    """System and environment utilities."""
    
    @staticmethod
    def get_dependency_versions() -> Dict[str, str]:
        """Return versions of key runtime dependencies."""
        return { 
            "numpy": np.__version__,
            "scipy": sp.__version__,
            "OpenGL": getattr(OpenGL, "__version__", "unknown"),
        }
    
    @staticmethod
    def configure_logging(level: int = logging.INFO) -> None:
        """Configure root logger for the application."""
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class Utilities(MathUtils):
    """Deprecated: Use specific utility classes instead."""
    pass



        



    


