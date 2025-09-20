from __future__ import annotations

"""
2D Black hole simulation entry point.

This module sets up core scientific and OpenGL imports. Subsequent
simulation code will build on these foundations.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable

import numpy as np
import scipy as sp
from scipy import constants as phys
from scipy.integrate import solve_ivp
from OpenGL import GL as gl  # GL API (functions/constants)
from OpenGL.GL.shaders import compileProgram, compileShader
import OpenGL  # Only used for package version metadata
import glfw
import time

from vector import Vec3, Vec4, Vec2

# Physical constants
c = phys.c  # Speed of light (m/s)
G = phys.G  # Gravitational constant (m³/kg⋅s²)

from utilities import SystemUtils

@dataclass
class Engine:
    """OpenGL rendering engine for the black hole simulation."""
    
    # Window dimensions in pixels
    window_width: int = 800
    window_height: int = 600

    # Physical simulation space dimensions (meters)
    sim_width: float = 1.0e11   # ~100 billion meters  
    sim_height: float = 7.5e10  # ~75 billion meters

    # Navigation state for camera control
    offset_x: float = 0.0      # Camera X offset (meters)
    offset_y: float = 0.0      # Camera Y offset (meters)
    zoom: float = 1.0          # Zoom level

    # Mouse interaction state
    middle_mouse_down: bool = False
    last_mouse_x: float = 0.0
    last_mouse_y: float = 0.0

    window: Optional[object] = None

    def __init__(self) -> None:
        """Initialize GLFW window and OpenGL context."""
        logger = logging.getLogger("blackhole.engine")
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("GLFW initialization failed")

        self.window = glfw.create_window(
            self.window_width, self.window_height, "Black Hole Simulation", None, None
        )
        if not self.window:
            logger.error("Failed to create GLFW window")
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # enable vsync if available

        # Initial GL state
        self._setup_viewport_and_projection(self.window_width, self.window_height)
        gl.glClearColor(0.05, 0.07, 0.10, 1.0)

        # Callbacks
        glfw.set_window_size_callback(self.window, self.on_resize)
        glfw.set_key_callback(self.window, self.on_key)

        logger.info("Created window %dx%d px", self.window_width, self.window_height)

    def _setup_viewport_and_projection(self, width_px: int, height_px: int) -> None:
        """Setup OpenGL viewport and basic orthographic projection."""
        gl.glViewport(0, 0, width_px, height_px)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, width_px, 0, height_px, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def on_resize(self, _window, width: int, height: int) -> None:
        """Handle window resize events."""
        self._setup_viewport_and_projection(width, height)

    def on_key(self, window, key, scancode, action, mods):  # noqa: D401 (GLFW signature)
        """Handle keyboard input - ESC to close."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def clear_and_setup_projection(self) -> None:
        """Clear screen and setup physics-based orthographic projection."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # Map physical space (meters) to screen coordinates
        left   = -self.sim_width + self.offset_x
        right  =  self.sim_width + self.offset_x
        bottom = -self.sim_height + self.offset_y
        top    =  self.sim_height + self.offset_y
        gl.glOrtho(left, right, bottom, top, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def render_frame(self) -> None:
        """Render a single frame of the simulation."""
        logger = logging.getLogger("blackhole.engine")        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        left   = -self.sim_width + self.offset_x
        right  =  self.sim_width + self.offset_x
        bottom = -self.sim_height + self.offset_y
        top    =  self.sim_height + self.offset_y
        gl.glOrtho(left, right, bottom, top, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glfw.swap_buffers(self.window)

    def run(self) -> None:
        """Main loop for the simulation."""
        logger = logging.getLogger("blackhole.engine")
        logger.info(
            "Starting main loop (window: %dx%d px)", self.window_width, self.window_height
        )
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        left   = -self.sim_width + self.offset_x
        right  =  self.sim_width + self.offset_x
        bottom = -self.sim_height + self.offset_y
        top    =  self.sim_height + self.offset_y
        gl.glOrtho(left, right, bottom, top, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        assert self.window is not None
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        while not glfw.window_should_close(self.window):
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 0.01:
                fps = frame_count / elapsed_time
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time
            glfw.poll_events()
            self.render_frame()
        logger.info("Window closed")

    def shutdown(self) -> None:
        """Clean up GLFW resources."""
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()

@dataclass
class BlackHole:
    """Schwarzschild black hole with event horizon rendering."""
    
    mass: float                    # Black hole mass (kg)
    position: Vec3                 # 3D position (meters)
    r_s: float = field(init=False) # Schwarzschild radius (computed)

    def __post_init__(self) -> None:
        """Validate inputs and compute Schwarzschild radius."""
        if not isinstance(self.position, Vec3):
            try:
                self.position = Vec3.from_iterable(self.position)  # type: ignore[arg-type]
            except Exception as exc:
                raise TypeError("position must be Vec3 or 3-element iterable") from exc
        # Schwarzschild radius: r_s = 2GM/c²
        self.r_s = 2.0 * phys.G * self.mass / (phys.c ** 2)
    
    def DrawBlackHole(self) -> None:
        """Render the black hole as a filled circle at the event horizon."""
        gl.glBegin(gl.GL_TRIANGLE_FAN)
        gl.glColor3f(1.0, 0.0, 0.0)  # Red color for visibility
        gl.glVertex2f(0.0, 0.0)  # Center at origin
        # Draw circle with radius = Schwarzschild radius
        for i in range(101):
            angle = 2.0 * math.pi * i / 100
            x = self.r_s * math.cos(angle)
            y = self.r_s * math.sin(angle)
            gl.glVertex2f(x, y)
        gl.glEnd()

@dataclass
class Ray:
    """Light ray following geodesics in Schwarzschild spacetime."""
    
    # Initial conditions
    pos: Vec2                      # Starting position (meters)
    dir: Vec2                      # Starting direction/velocity (m/s)
        
    # Polar coordinates for geodesic integration
    r: float = field(init=False)     # Radial distance from black hole
    dr: float = field(init=False)    # Radial velocity (dr/dλ)
    phi: float = field(init=False)   # Angular position
    dphi: float = field(init=False)  # Angular velocity (dφ/dλ)
    
    # Conserved quantities in Schwarzschild metric
    E: float = field(init=False)     # Energy per unit mass
    L: float = field(init=False)     # Angular momentum per unit mass
    
    # Visualization trail
    trail: list[Vec2] = field(default_factory=list)  # Path history
    max_trail_length: int = 500                      # Maximum trail points
    
    # Event horizon physics
    crossed_horizon: bool = False   # Has ray crossed r = r_s?
    
    def __post_init__(self) -> None:

        # Step 2: Convert to polar coordinates (r, phi)
        self.r = math.sqrt(self.pos.x * self.pos.x + self.pos.y * self.pos.y)
        self.phi = math.atan2(self.pos.y, self.pos.x)
        
        # Step 3: Transform direction from cartesian to polar velocities
        cos_phi = math.cos(self.phi)
        sin_phi = math.sin(self.phi)
        
        self.dr = self.dir.x * cos_phi + self.dir.y * sin_phi  # m/s
        self.dphi = (-self.dir.x * sin_phi + self.dir.y * cos_phi) / self.r  # rad/s
        
        # Step 4: Compute conserved quantities
        self.L = self.r * self.r * self.dphi
        f = 1.0 - SagA.r_s / self.r
        dt_dlambda = math.sqrt((self.dr * self.dr) / (f * f) + (self.r * self.r * self.dphi * self.dphi) / f)
        self.E = f * dt_dlambda

    def step(self, dLambda: float, r_s: float) -> None:
        """Step the ray forward using RK4 integration."""
        
        # Check if ray has crossed the event horizon
        if self.r <= r_s and not self.crossed_horizon:
            self.crossed_horizon = True
        
        # If ray has crossed horizon, fade out trail instead of stepping
        if self.crossed_horizon:
            # Remove points from trail to create fading effect
            if len(self.trail) > 0:
                # Remove multiple points per frame for faster fading
                points_to_remove = min(3, len(self.trail))
                for _ in range(points_to_remove):
                    self.trail.pop(0)
            return
            
        # Normal integration for rays that haven't crossed horizon
        rk4Step(self, dLambda, r_s)
        
        # Update cartesian position from polar coordinates
        self.pos.x = self.r * math.cos(self.phi)
        self.pos.y = self.r * math.sin(self.phi)
        
        # Add to trail for visualization
        self.trail.append(Vec2(self.pos.x, self.pos.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)
    
    def draw(self, rays: list['Ray']) -> None:
        """Render this ray using standalone drawing function."""
        DrawRay(self)
    
    def is_completely_faded(self) -> bool:
        """Check if ray has crossed horizon and trail is completely gone."""
        return self.crossed_horizon and len(self.trail) == 0

def DrawRays(rays: list[Ray]) -> None:
    """Render all rays by calling DrawRay"""
    for ray in rays:
        DrawRay(ray)

def DrawRay(ray: Ray) -> None:
    """Render single ray with red position dot and white fading trail."""
    # Only draw current position point if ray hasn't crossed horizon
    if not ray.crossed_horizon:
        gl.glPointSize(3.0)
        gl.glColor3f(1.0, 0.0, 0.0)  # Red point for current position
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex2f(ray.pos.x, ray.pos.y)
        gl.glEnd()
    
    # Draw trail with alpha fading (old=transparent, new=opaque)
    N = len(ray.trail)
    if N < 2:
        return
        
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glLineWidth(1.0)
    
    gl.glBegin(gl.GL_LINE_STRIP)
    for i, point in enumerate(ray.trail):
        alpha = float(i) / float(N - 1)  # 0.0 (old) to 1.0 (new)
        alpha = max(alpha, 0.05)         # Minimum visibility
        gl.glColor4f(1.0, 1.0, 1.0, alpha)  # White with alpha
        gl.glVertex2f(point.x, point.y)
    gl.glEnd()
    
    gl.glDisable(gl.GL_BLEND)

def GeodesicRHS(ray: Ray, rhs: np.array(4), r_s: float) -> None:
    """Compute right-hand side of geodesic equations in Schwarzschild metric."""
    f = 1.0 - r_s/ray.r                    # Metric coefficient
    dt_dLambda = ray.E/f                   # Time coordinate velocity

    # Geodesic equations: d²x^μ/dλ² + Γ^μ_αβ dx^α/dλ dx^β/dλ = 0
    rhs[0] = ray.dr                        # dr/dλ
    rhs[1] = ray.dphi                      # dφ/dλ
    rhs[2] = -(r_s/(2.0*ray.r*ray.r)) * f * (dt_dLambda*dt_dLambda) + (r_s/(2.0*ray.r*ray.r*f)) * (ray.dr*ray.dr) + (ray.r - r_s) * (ray.dphi*ray.dphi)  # d²r/dλ²
    rhs[3] = -2.0 * ray.dr * ray.dphi / ray.r  # d²φ/dλ²

    return rhs

def AddState(a: np.array(4), b: np.array(4), factor: float) -> np.array(4):
    """Add scaled vector b to vector a: a += factor * b."""
    for i in range(4):
        a[i] += factor * b[i]
    return a

def rk4Step(ray: Ray, dLambda: float, r_s: float) -> None:
    """
    Improved Runge-Kutta 4th order step with event horizon checking.
    Uses adaptive step reduction when intermediate steps cross the event horizon.
    """
    # State vector: [r, phi, dr, dphi]
    y0 = np.array([ray.r, ray.phi, ray.dr, ray.dphi])
    
    # Pre-allocate k arrays for RK4 coefficients
    k1 = np.zeros(4)
    k2 = np.zeros(4)
    k3 = np.zeros(4)
    k4 = np.zeros(4)
    temp = np.zeros(4)

    # First stage: k1
    GeodesicRHS(ray, k1, r_s)
    temp = AddState(y0.copy(), k1, dLambda/2.0)
    
    # Check if intermediate step crosses event horizon
    if temp[0] <= r_s* 1.01:  # r2.r <= rs
        # Use Euler step instead of full RK4
        ray.r += dLambda * k1[0]
        ray.phi += dLambda * k1[1]
        ray.dr += dLambda * k1[2]
        ray.dphi += dLambda * k1[3]
        return

    # Second stage: k2
    # Create temporary ray for k2 calculation without triggering __post_init__
    r2_r, r2_phi, r2_dr, r2_dphi = temp[0], temp[1], temp[2], temp[3]
    tempRay = object.__new__(Ray)  # Create without calling __init__
    tempRay.r = r2_r
    tempRay.phi = r2_phi
    tempRay.dr = r2_dr
    tempRay.dphi = r2_dphi
    tempRay.E = ray.E  # Copy conserved quantities
    tempRay.L = ray.L
    
    GeodesicRHS(tempRay, k2, r_s)
    temp = AddState(y0.copy(), k2, dLambda/2.0)
    
    # Check if intermediate step crosses event horizon
    if temp[0] <= r_s*1.01:  # r3.r <= rs
        # Use partial RK2 step
        ray.r += (dLambda / 3.0) * (k1[0] + 2 * k2[0])
        ray.phi += (dLambda / 3.0) * (k1[1] + 2 * k2[1])
        ray.dr += (dLambda / 3.0) * (k1[2] + 2 * k2[2])
        ray.dphi += (dLambda / 3.0) * (k1[3] + 2 * k2[3])
        return

    # Third stage: k3
    r3_r, r3_phi, r3_dr, r3_dphi = temp[0], temp[1], temp[2], temp[3]
    tempRay.r = r3_r
    tempRay.phi = r3_phi
    tempRay.dr = r3_dr
    tempRay.dphi = r3_dphi
    
    GeodesicRHS(tempRay, k3, r_s)
    temp = AddState(y0.copy(), k3, dLambda)
    
    # Check if intermediate step crosses event horizon
    if temp[0] <= r_s*1.01:  # r4.r <= rs
        # Use partial RK3 step
        ray.r += (dLambda / 5.0) * (k1[0] + 2 * k2[0] + 2 * k3[0])
        ray.phi += (dLambda / 5.0) * (k1[1] + 2 * k2[1] + 2 * k3[1])
        ray.dr += (dLambda / 5.0) * (k1[2] + 2 * k2[2] + 2 * k3[2])
        ray.dphi += (dLambda / 5.0) * (k1[3] + 2 * k2[3] + 2 * k3[3])
        return

    # Fourth stage: k4
    r4_r, r4_phi, r4_dr, r4_dphi = temp[0], temp[1], temp[2], temp[3]
    tempRay.r = r4_r
    tempRay.phi = r4_phi
    tempRay.dr = r4_dr
    tempRay.dphi = r4_dphi
    
    GeodesicRHS(tempRay, k4, r_s)

    # Full RK4 step (no intermediate crossings detected)
    ray.r += (dLambda/6.0) * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0])
    ray.phi += (dLambda/6.0) * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1])
    ray.dr += (dLambda/6.0) * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2])
    ray.dphi += (dLambda/6.0) * (k1[3] + 2.0*k2[3] + 2.0*k3[3] + k4[3])
 
def initialize_rays_random(count: int):
    """Create rays with random positions and velocities for varied trajectories."""
    import random
    
    random.seed(42)  # Reproducible results
    
    for i in range(count):
        # Starting distance: 20-100 billion meters from black hole
        start_distance = random.uniform(2e10, 1e11)
        
        # Random starting angle
        start_angle = random.uniform(0, 2 * math.pi)
        
        # Convert to cartesian coordinates
        start_x = start_distance * math.cos(start_angle)
        start_y = start_distance * math.sin(start_angle)
        initial_pos = Vec2(start_x, start_y)
        
        # Velocity: 30-95% speed of light
        vel_magnitude = c * random.uniform(0.3, 0.95)
        
        # Mostly tangential velocity with some randomness
        tangent_angle = start_angle + math.pi/2  # Perpendicular to radial
        vel_angle = tangent_angle + random.uniform(-math.pi/4, math.pi/4)
        
        vel_x = vel_magnitude * math.cos(vel_angle)
        vel_y = vel_magnitude * math.sin(vel_angle)
        initial_dir = Vec2(vel_x, vel_y)
        
        Rays.append(Ray(initial_pos, initial_dir))

def initialize_rays_uniform(count: int):
    """Create parallel rays for testing gravitational lensing effects."""
    for i in range(count):
        # Start rays in a vertical line on the left side
        initial_pos = Vec2(-1e11, 3.27606302719999999e10*(2*i/count))
        
        # All rays moving horizontally at 95% speed of light
        vel_magnitude = c * 0.95
        vel_x = vel_magnitude  # Rightward velocity
        vel_y = 0.0            # No vertical component
        initial_dir = Vec2(vel_x, vel_y)
        
        Rays.append(Ray(initial_pos, initial_dir))

# Global simulation objects
SagA = BlackHole(8.54e36, Vec3(0, 0, 0))  # Sagittarius A* (4.31M solar masses)
Rays: list[Ray] = []                       # List of light rays to simulate

def main():
    """Main simulation loop"""
    SystemUtils.configure_logging()
    logger = logging.getLogger("blackhole")
    logger.info("Dependency versions: %s", SystemUtils.get_dependency_versions())
    
    # Initialize rays (choose initialization method)
    #initialize_rays_random(100)
    initialize_rays_uniform(100)
    
    engine = Engine()
    try:
        # FPS monitoring variables
        frame_count = 0
        start_time = time.time()
        
        # Main simulation loop
        while not glfw.window_should_close(engine.window):
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Log FPS every 0.5 seconds
            if elapsed_time >= 0.5:
                fps = frame_count / elapsed_time
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time
            
            # Clear screen and setup physics-based projection
            engine.clear_and_setup_projection()
            
            # Render black hole event horizon
            SagA.DrawBlackHole()
            
            # Update and render each ray
            for ray in Rays:
                ray.step(1.0, SagA.r_s)  # Integrate geodesic (dλ = 1.0)
                ray.draw(Rays)           # Render ray trail
            
            # Present frame and handle input
            glfw.swap_buffers(engine.window)
            glfw.poll_events()
    finally:
        engine.shutdown()

if __name__ == "__main__":
    main()
