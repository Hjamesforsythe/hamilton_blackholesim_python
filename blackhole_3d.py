from __future__ import annotations

"""
3D Black hole simulation with GPU-accelerated raytracing.

This module implements a modern OpenGL-based black hole visualization with:
- Compute shader-based geodesic raytracing (when supported)
- VBO/VAO mesh system for warped spacetime grid
- UBO (Uniform Buffer Objects) for efficient GPU data transfer
- Fallback CPU rendering for older hardware

GPU Features:
- OpenGL 4.3+ Core Profile with compute shader support
- Dynamic resolution based on camera movement
- External shader file loading (grid.vert, grid.frag, geodesic.comp)
- Schwarzschild geometry visualization
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy as sp
from scipy import constants as phys
from OpenGL import GL as gl  # GL API (functions/constants)
from OpenGL.GL.shaders import compileProgram, compileShader
import OpenGL  # Only used for package version metadata
import glfw 
import time

from vector import Vec3, Vec4
from utilities import MathUtils, MatrixUtils, SystemUtils, ShaderUtils

# Constants
C = phys.c
G = phys.G
Pi = math.pi

@dataclass
class Camera:
    """Camera class for 3D navigation around the black hole."""
    target: Vec3  # Center the camera orbit on the black hole (0,0,0)
    radius: float = 9.3419e10
    minradius: float = 1e10
    maxradius: float = 1e12
    fov: float = 45.0
    azimuth: float = 0.0
    elevation: float = 0.0

    orbitSpeed: float = 0.01
    zoomSpeed: float = 0.01

    dragging: bool = False
    panning: bool = False
    moving: bool = False

    LastX: float = 0.0
    LastY: float = 0.0

    def __init__(self) -> None:
        self.target = Vec3(0, 0, 0)

    #get the camera position
    def getPosition(self) -> Vec3:
        #calculate the camera position
        clampedElevation: float = MathUtils.clamp(self.elevation, 0.01, Pi - 0.01)

        return Vec3(
            self.radius * math.sin(clampedElevation) * math.cos(self.azimuth),
            self.radius * math.cos(clampedElevation),
            self.radius * math.sin(clampedElevation) * math.sin(self.azimuth)
        )

    def update(self) -> None:
        target: Vec3 = self.target
        position: Vec3 = self.getPosition()
        if(self.dragging or self.panning):
            self.moving = True
        else:
            self.moving = False

    def processMouseMove(self, window, xpos: float, ypos: float) -> None:
        """Handle mouse movement for camera orbit controls."""
        dx: float = xpos - self.LastX
        dy: float = ypos - self.LastY

        if self.dragging and not self.panning:
            # Orbit around target
            self.azimuth += dx * self.orbitSpeed
            self.elevation -= dy * self.orbitSpeed
            self.elevation = MathUtils.clamp(self.elevation, 0.01, Pi - 0.01)
        
        self.LastX = xpos
        self.LastY = ypos
        self.update()
    
    def processMouseButton(self, window, button: int, action: int, mods: int) -> None:
        """Handle mouse button events for camera controls."""
        if button == glfw.MOUSE_BUTTON_LEFT or button == glfw.MOUSE_BUTTON_MIDDLE:
            if action == glfw.PRESS:
                self.dragging = True
                # Disable panning so camera always orbits center
                self.panning = False
                self.LastX, self.LastY = glfw.get_cursor_pos(window)
            elif action == glfw.RELEASE:
                self.dragging = False
                self.panning = False
    
    def processScroll(self, window, xoffset: float, yoffset: float) -> None:
        """Handle mouse scroll for camera zoom."""
        self.radius -= yoffset * self.zoomSpeed
        self.radius = MathUtils.clamp(self.radius, self.minradius, self.maxradius)
        self.update()
    
    def processKey(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """Handle keyboard input for camera controls."""
        pass  # No keyboard controls currently implemented
    
    def on_resize(self, window, width: int, height: int) -> None:
        """Handle window resize events."""
        pass  # Will be implemented when Engine is complete
    
    def on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        """Handle keyboard events."""
        self.processKey(window, key, scancode, action, mods) 


@dataclass
class Engine:
    """OpenGL rendering engine for the black hole simulation."""

    # Window dimensions in pixels
    window_width: int = 800
    window_height: int = 600

    # Physical simulation space dimensions (meters)
    sim_width: float = 1.0e11   # ~100 billion meters  
    sim_height: float = 7.5e10  # ~75 billion meters

    compute_width: int = 200   
    compute_height: int = 150  

    # GPU resources
    quad_vao: Optional[int] = None
    texture: Optional[int] = None
    shader_program: Optional[int] = None
    compute_program: Optional[int] = None
    
    # UBOs (Uniform Buffer Objects)
    camera_ubo: Optional[int] = None
    disk_ubo: Optional[int] = None
    objects_ubo: Optional[int] = None
    
    # Grid mesh resources
    grid_vao: Optional[int] = None
    grid_vbo: Optional[int] = None
    grid_ebo: Optional[int] = None
    grid_index_count: int = 0
    grid_shader_program: Optional[int] = None
    grid_generated: bool = False  # Track if grid is already generated
    
    # Cached matrices for performance
    cached_view_matrix: Optional[np.ndarray] = None
    cached_proj_matrix: Optional[np.ndarray] = None
    cached_view_proj_matrix: Optional[np.ndarray] = None
    last_camera_pos: Optional[Vec3] = None
    last_window_aspect: float = 0.0
    
    # UBO upload tracking for performance
    last_camera_upload_pos: Optional[Vec3] = None
    last_camera_moving: bool = False
    disk_ubo_uploaded: bool = False
    objects_ubo_uploaded: bool = False

    def __init__(self) -> None:
        """Initialize GLFW window and OpenGL context."""
        logger = logging.getLogger("blackhole.engine")
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("GLFW initialization failed")

        # Set OpenGL context version - use Core Profile 4.3+ for compute shader support
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(
            self.window_width, self.window_height, "Black Hole Simulation 3D", None, None
            )
        
        if not self.window:
            logger.error("Failed to create GLFW window")
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # enable vsync if available

        # Initial GL state
                # Setup OpenGL state
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.05, 0.07, 0.10, 1.0)
        
        # Setup viewport and projection
        self._setup_viewport_and_projection(self.window_width, self.window_height)
        
        # Set callbacks
        glfw.set_window_size_callback(self.window, self.on_resize)
        glfw.set_key_callback(self.window, self.on_key)

        # Initialize GPU resources
        self._initialize_shaders()
        if self.compute_program is not None:
            self._initialize_ubos()
        self._initialize_quad()

        logger.info("Created 3D window %dx%d px", self.window_width, self.window_height)

    def _setup_viewport_and_projection(self, width_px: int, height_px: int) -> None:
        """Setup OpenGL viewport and 3D perspective projection."""
        gl.glViewport(0, 0, width_px, height_px)
        
    def on_resize(self, _window, width: int, height: int) -> None:
        """Handle window resize events."""
        self.window_width = width
        self.window_height = height
        self._setup_viewport_and_projection(width, height)

    def on_key(self, window, key, scancode, action, mods) -> None:
        """Handle keyboard input - ESC to close."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)


    def _initialize_shaders(self) -> None:
        """Initialize shader programs."""
        # Create basic vertex/fragment shader for fullscreen quad
        self.shader_program = self._create_shader_program()
        
        # Load grid shader program
        try:
            self.grid_shader_program = self._create_grid_shader_program()
            print("✓ Grid shaders loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load grid shaders: {e}")
            print(f"  Error details: {type(e).__name__}: {str(e)}")
            self.grid_shader_program = None
        
        # Check if compute shaders are supported (requires OpenGL 4.3+)
        major = gl.glGetIntegerv(gl.GL_MAJOR_VERSION)
        minor = gl.glGetIntegerv(gl.GL_MINOR_VERSION)
        
        if major > 4 or (major == 4 and minor >= 3):
            try:
                # Load compute shader for geodesic raytracing
                self.compute_program = self._create_compute_program("shaders/geodesic.comp")
                print("✓ Compute shader loaded successfully")
            except Exception as e:
                print(f"⚠ Failed to load compute shader: {e}")
                print("  Falling back to CPU-based rendering")
                self.compute_program = None
        else:
            print(f"⚠ OpenGL {major}.{minor} detected - Compute shaders require 4.3+")
            print("  Falling back to CPU-based rendering")
            self.compute_program = None
    
    def _initialize_ubos(self) -> None:
        """Initialize Uniform Buffer Objects."""
        # Camera UBO (binding = 1)
        self.camera_ubo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.camera_ubo)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 128, None, gl.GL_DYNAMIC_DRAW)  # ~128 bytes
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 1, self.camera_ubo)
        
        # Disk UBO (binding = 2)
        self.disk_ubo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.disk_ubo)
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, 16, None, gl.GL_DYNAMIC_DRAW)  # 4 floats
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 2, self.disk_ubo)
        
        # Objects UBO (binding = 3)
        self.objects_ubo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.objects_ubo)
        # Space for: int numObjects + padding + 16×(vec4 posRadius + vec4 color + vec4 mass_padded)
        ubo_size = 16 + 16 * (16 + 16 + 16)  # 16 + 16*48 = 784 bytes
        gl.glBufferData(gl.GL_UNIFORM_BUFFER, ubo_size, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_UNIFORM_BUFFER, 3, self.objects_ubo)
    
    def _initialize_quad(self) -> None:
        """Initialize fullscreen quad for displaying compute shader output."""
        # Fullscreen quad vertices (position + texcoord)
        quad_vertices = np.array([
            # positions    # texCoords
            -1.0,  1.0,    0.0, 1.0,  # top left
            -1.0, -1.0,    0.0, 0.0,  # bottom left
             1.0, -1.0,    1.0, 0.0,  # bottom right
            -1.0,  1.0,    0.0, 1.0,  # top left
             1.0, -1.0,    1.0, 0.0,  # bottom right
             1.0,  1.0,    1.0, 1.0   # top right
        ], dtype=np.float32)
        
        # Create VAO and VBO
        self.quad_vao = gl.glGenVertexArrays(1)
        quad_vbo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.quad_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, gl.GL_STATIC_DRAW)
        
        # Position attribute (location = 0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, None)
        gl.glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute (location = 1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * 4, gl.ctypes.c_void_p(2 * 4))
        gl.glEnableVertexAttribArray(1)
        
        # Create texture for compute shader output
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, self.compute_width, 
                       self.compute_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        
        gl.glBindVertexArray(0)
    
    def _create_shader_program(self) -> int:
        """Create basic vertex/fragment shader program for fullscreen quad."""
        vertex_source = """
        #version 430 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
        """
        
        fragment_source = """
        #version 430 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main() {
            FragColor = texture(screenTexture, TexCoord);
        }
        """
        
        # Compile shaders
        vertex_shader = compileShader(vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_source, gl.GL_FRAGMENT_SHADER)
        
        # Link program
        program = compileProgram(vertex_shader, fragment_shader)
        
        return program
    
    def _create_shader_from_file(self, vertex_path: str, fragment_path: str) -> int:
        """Create shader program from external files."""
        return ShaderUtils.create_shader_program(vertex_path, fragment_path)
    
    def _create_compute_program(self, shader_path: str) -> int:
        """Load and compile compute shader program."""
        return ShaderUtils.create_compute_program(shader_path)
    
    def upload_camera_ubo(self, camera: Camera) -> None:
        """Upload camera data to UBO only if changed."""
        import struct
        
        # Check if camera data needs updating
        cam_pos = camera.getPosition()
        if (self.last_camera_upload_pos is not None and 
            abs(cam_pos.x - self.last_camera_upload_pos.x) < 1e6 and  # 1km threshold
            abs(cam_pos.y - self.last_camera_upload_pos.y) < 1e6 and
            abs(cam_pos.z - self.last_camera_upload_pos.z) < 1e6 and
            self.last_camera_moving == camera.moving):
            return  # No significant change, skip upload
        
        # Calculate camera vectors
        target = camera.target
        
        # Calculate view vectors
        forward = Vec3(target.x - cam_pos.x, target.y - cam_pos.y, target.z - cam_pos.z).normalized()
        up = Vec3(0, 1, 0)  # World up
        right = forward.cross(up).normalized()
        up = right.cross(forward).normalized()
        
        # Camera parameters
        tan_half_fov = math.tan(math.radians(60.0 * 0.5))  # 60 degree FOV
        aspect = float(self.window_width) / float(self.window_height)
        
        # Pack data according to std140 layout (matches compute shader)
        data = struct.pack(
            '18f2i',  # 18 floats + 2 ints = 20 total items
            cam_pos.x, cam_pos.y, cam_pos.z, 0.0,      # vec3 pos + padding
            right.x, right.y, right.z, 0.0,            # vec3 right + padding
            up.x, up.y, up.z, 0.0,                     # vec3 up + padding
            forward.x, forward.y, forward.z, 0.0,      # vec3 forward + padding
            tan_half_fov, aspect,                       # float tanHalfFov, float aspect
            1 if camera.moving else 0,                  # bool moving (as int)
            0                                           # int _pad4
        )
        
        # Upload to GPU
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.camera_ubo)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, len(data), data)
        
        # Update tracking variables
        self.last_camera_upload_pos = Vec3(cam_pos.x, cam_pos.y, cam_pos.z)
        self.last_camera_moving = camera.moving
    
    def upload_disk_ubo(self) -> None:
        """Upload disk parameters to UBO (only once - data is static)."""
        if self.disk_ubo_uploaded:
            return  # Disk data is static, no need to re-upload
            
        import struct
        
        # Disk parameters (matching compute shader)
        r1 = SagA.r_s * 2.2  # Inner radius
        r2 = SagA.r_s * 5.2  # Outer radius
        num_rays = 2.0
        thickness = 1e9
        
        data = struct.pack('4f', r1, r2, num_rays, thickness)
        
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.disk_ubo)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, len(data), data)
        self.disk_ubo_uploaded = True
    
    def upload_objects_ubo(self, objects_list: list) -> None:
        """Upload objects data to UBO (only once - data is static)."""
        if self.objects_ubo_uploaded:
            return  # Objects data is static, no need to re-upload
            
        import struct
        
        num_objects = min(len(objects_list), 16)  # Max 16 objects
        
        # Pack number of objects (with padding for std140 - int takes 16 bytes alignment)
        data = struct.pack('4i', num_objects, 0, 0, 0)
        
        # Pack vec4 objPosRadius[16] - each vec4 is 16 bytes
        for i in range(16):
            if i < len(objects_list):
                obj = objects_list[i]
                data += struct.pack('4f', 
                    obj.position_radius.x, obj.position_radius.y, 
                    obj.position_radius.z, obj.position_radius.w)
            else:
                data += struct.pack('4f', 0.0, 0.0, 0.0, 0.0)
        
        # Pack vec4 objColor[16] - each vec4 is 16 bytes  
        for i in range(16):
            if i < len(objects_list):
                obj = objects_list[i]
                data += struct.pack('4f',
                    obj.color.x, obj.color.y, obj.color.z, obj.color.w)
            else:
                data += struct.pack('4f', 0.0, 0.0, 0.0, 0.0)
        
        # Pack float mass[16] - each float in array takes 16 bytes in std140
        for i in range(16):
            if i < len(objects_list):
                obj = objects_list[i]
                data += struct.pack('4f', obj.mass, 0.0, 0.0, 0.0)
            else:
                data += struct.pack('4f', 0.0, 0.0, 0.0, 0.0)
        
        gl.glBindBuffer(gl.GL_UNIFORM_BUFFER, self.objects_ubo)
        gl.glBufferSubData(gl.GL_UNIFORM_BUFFER, 0, len(data), data)
        self.objects_ubo_uploaded = True
    
    def dispatch_compute(self, camera: Camera) -> None:
        """Dispatch compute shader for raytracing or fallback to CPU."""
        if self.compute_program is None:
            # Fallback: Create a simple colored texture
            self._create_fallback_texture()
            return
            
        # Determine compute resolution based on camera movement
        compute_w = self.compute_width if camera.moving else 200
        compute_h = self.compute_height if camera.moving else 150
        
        # Reallocate texture if needed
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, compute_w, compute_h,
                       0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        
        # Use compute program and upload UBOs
        gl.glUseProgram(self.compute_program)
        self.upload_camera_ubo(camera)
        self.upload_disk_ubo()
        self.upload_objects_ubo(objects)
        
        # Bind texture as image
        gl.glBindImageTexture(0, self.texture, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_RGBA8)
        
        # Dispatch compute shader
        groups_x = (compute_w + 15) // 16  # 16x16 local work group size
        groups_y = (compute_h + 15) // 16
        gl.glDispatchCompute(groups_x, groups_y, 1)
        
        # Memory barrier
        gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    
    def _create_fallback_texture(self) -> None:
        """Create a fallback texture when compute shaders aren't available."""
        # Create a simple gradient texture as placeholder
        width, height = 200, 150
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Create a simple radial gradient
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx*dx + dy*dy) / max(width, height)
                
                # Create a space-like gradient
                r = int(dist * 50)
                g = int(dist * 20) 
                b = int(dist * 80)
                pixels[y, x] = [r, g, b, 255]
        
        # Upload to texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height,
                       0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels)
    
    def draw_fullscreen_quad(self) -> None:
        """Draw fullscreen quad with compute shader output."""
        gl.glUseProgram(self.shader_program)
        gl.glBindVertexArray(self.quad_vao)
        
        # Bind texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "screenTexture"), 0)
        
        # Disable depth test for background
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glEnable(gl.GL_DEPTH_TEST)
    
    def generate_grid(self, objects_list: list) -> None:
        """Generate warped grid mesh based on Schwarzschild geometry."""
        # Only generate once - grid is static for given objects
        if self.grid_generated:
            return
            
        grid_size = 10  # Smaller grid for debugging
        spacing = 5e9   # Smaller spacing for debugging
        
        vertices = []
        indices = []
        
        # Generate grid vertices
        for z in range(grid_size + 1):
            for x in range(grid_size + 1):
                world_x = (x - grid_size // 2) * spacing
                world_z = (z - grid_size // 2) * spacing
                
                y = 0.0  # Flat grid for debugging - no warping
                
                # TODO: Re-enable Schwarzschild warping once basic grid works
                # Temporarily disabled for debugging
                
                vertices.extend([world_x, y, world_z])
        
        # Generate indices for line rendering
        for z in range(grid_size):
            for x in range(grid_size):
                i = z * (grid_size + 1) + x
                # Horizontal lines
                indices.extend([i, i + 1])
                # Vertical lines  
                indices.extend([i, i + grid_size + 1])
        
        # Convert to numpy arrays
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Upload to GPU
        if self.grid_vao is None:
            self.grid_vao = gl.glGenVertexArrays(1)
        if self.grid_vbo is None:
            self.grid_vbo = gl.glGenBuffers(1)
        if self.grid_ebo is None:
            self.grid_ebo = gl.glGenBuffers(1)
        
        gl.glBindVertexArray(self.grid_vao)
        
        # Upload vertex data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.grid_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_DYNAMIC_DRAW)
        
        # Upload index data
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.grid_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        # Setup vertex attributes
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
        
        self.grid_index_count = len(indices)
        self.grid_generated = True  # Mark as generated
        print(f"✓ Grid generated: {len(vertices)//3} vertices, {len(indices)} indices")
        print(f"  Grid extent: {grid_size * spacing:.2e} meters")
        print(f"  Grid spacing: {spacing:.2e} meters")
        gl.glBindVertexArray(0)
    
    def draw_grid(self, view_proj_matrix: np.ndarray) -> None:
        """Draw the warped grid mesh."""
        if self.grid_shader_program is None:
            print("⚠ Grid shader program not available - skipping grid rendering")
            return
        
        if self.grid_index_count == 0:
            print("⚠ Grid not generated yet - skipping grid rendering")
            return
        
        gl.glUseProgram(self.grid_shader_program)
        
        # Upload view-projection matrix
        loc = gl.glGetUniformLocation(self.grid_shader_program, "viewProj")
        gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE, view_proj_matrix.astype(np.float32))
        
        gl.glBindVertexArray(self.grid_vao)
        
        # Enable blending for grid transparency
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw grid lines  
        gl.glDrawElements(gl.GL_LINES, self.grid_index_count, gl.GL_UNSIGNED_INT, None)
        
        # Restore state
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
    
    def _create_grid_shader_program(self) -> int:
        """Create shader program for grid rendering using external shader files."""
        return self._create_shader_from_file("shaders/grid.vert", "shaders/grid.frag")
    
    def get_view_proj_matrix(self, camera: Camera) -> np.ndarray:
        """Get cached view-projection matrix, recalculate only if camera moved."""
        current_pos = camera.getPosition()
        current_aspect = self.window_width / self.window_height
        
        # Check if we need to recalculate
        need_update = (
            self.cached_view_proj_matrix is None or
            self.last_camera_pos is None or
            abs(current_pos.x - self.last_camera_pos.x) > 1e6 or  # 1km threshold
            abs(current_pos.y - self.last_camera_pos.y) > 1e6 or
            abs(current_pos.z - self.last_camera_pos.z) > 1e6 or
            abs(current_aspect - self.last_window_aspect) > 0.01
        )
        
        if need_update:
            # Recalculate matrices
            self.cached_view_matrix = MatrixUtils.create_look_at_matrix(current_pos, camera.target, Vec3(0, 1, 0))
            self.cached_proj_matrix = MatrixUtils.create_perspective_matrix(60.0, current_aspect, 1e9, 1e14)
            self.cached_view_proj_matrix = self.cached_proj_matrix @ self.cached_view_matrix
            
            # Update cached values
            self.last_camera_pos = Vec3(current_pos.x, current_pos.y, current_pos.z)
            self.last_window_aspect = current_aspect
        
        return self.cached_view_proj_matrix

    def shutdown(self) -> None:
        """Clean up GLFW resources."""
        if self.window is not None:
            glfw.destroy_window(self.window)
        glfw.terminate()

@dataclass
class BlackHole:
    mass: float                    # Black hole mass (kg)
    position: Vec3                 # 3D position (meters)
    r_s: float = field(init=False) # Schwarzschild radius (computed)
    radius: float = 10.0

    def __post_init__(self) -> None:
        """Validate inputs and compute Schwarzschild radius."""
        if not isinstance(self.position, Vec3):
            try:
                self.position = Vec3.from_iterable(self.position)  # type: ignore[arg-type]
            except Exception as exc:
                raise TypeError("position must be Vec3 or 3-element iterable") from exc
        # Schwarzschild radius: r_s = 2GM/c²
        self.r_s = 2.0 * phys.G * self.mass / (phys.c ** 2)
    
    def intercept(self, px: float, py:float, pz:float)-> bool:
        """Check if a point is inside the black hole."""
        # we buffer everything by 1% to account for floating point precision
        dx: float = px - self.position.x
        dy: float = py - self.position.y
        dz: float = pz - self.position.z
        r2: float = dx * dx + dy * dy + dz * dz
        return r2 < (self.r_s * self.r_s * 1.01)


SagA = BlackHole(8.54e36, Vec3(0, 0, 0))

@dataclass
class ObjectData:
    position_radius: Vec4
    color: Vec4
    mass: float
    velocity: Vec3

objects = [
    ObjectData(
        position_radius=Vec4(4e11, 0.0, 0.0, 4e10),
        color=Vec4(1, 1, 0, 1),
        mass=1.98892e30,
        velocity=Vec3(0,0,0)
    ),
    ObjectData(
        position_radius=Vec4(0.0, 0.0, 4e11, 4e10),
        color=Vec4(1, 0, 0, 1),
        mass=1.98892e30,
        velocity=Vec3(0,0,0)
    ),
    ObjectData(
        position_radius=Vec4(0.0, 0.0, 0.0, SagA.r_s),
        color=Vec4(0, 0, 0, 1),
        mass=SagA.mass,
        velocity=Vec3(0,0,0)
    )
]

def setupCameraCallbacks(window, camera: Camera) -> None:
    glfw.set_window_user_pointer(window, camera)
    glfw.set_mouse_button_callback(window, camera.processMouseButton)
    glfw.set_cursor_pos_callback(window, camera.processMouseMove)
    glfw.set_scroll_callback(window, camera.processScroll)
    glfw.set_key_callback(window, camera.processKey)


def main() -> None:
    print("=== Black Hole Simulation 3D Starting ===")
    SystemUtils.configure_logging()
    logger = logging.getLogger("blackhole")
    logger.info("Dependency versions: %s", SystemUtils.get_dependency_versions())
    
    print("✓ Logging configured")
    print("✓ Dependencies loaded:", SystemUtils.get_dependency_versions())

    print("\n--- Initializing Engine ---")
    engine = Engine()
    print("✓ OpenGL Engine initialized")
    print(f"  Window size: {engine.window_width}x{engine.window_height} pixels")
    print(f"  Simulation space: {engine.sim_width:.2e} x {engine.sim_height:.2e} meters")
    
    print("\n--- Initializing Camera ---")
    camera = Camera()
    print("✓ Camera initialized")
    print(f"  Initial position: {camera.getPosition()}")
    print(f"  Target: {camera.target}")
    print(f"  Radius: {camera.radius:.2e} meters")
    print(f"  FOV: {camera.fov}°")

    print("\n--- Setting up Black Hole ---")
    print(f"✓ Black hole (Sgr A*) created:")
    print(f"  Mass: {SagA.mass:.2e} kg")
    print(f"  Position: {SagA.position}")
    print(f"  Schwarzschild radius: {SagA.r_s:.2e} meters")
    
    print("\n--- Setting up Objects ---")
    print(f"✓ {len(objects)} objects created:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: mass={obj.mass:.2e} kg, pos=({obj.position_radius.x:.2e}, {obj.position_radius.y:.2e}, {obj.position_radius.z:.2e})")

    setupCameraCallbacks(engine.window, camera)
    print("✓ Camera callbacks configured")
    
    print("\n--- Controls ---")
    print("  Left Mouse: Orbit camera around black hole")
    print("  Scroll: Zoom in/out")
    print("  ESC: Exit simulation")
    
    print("\n=== Starting Main Loop ===")
    print("Rendering black hole visualization...")

    try:
        # FPS monitoring variables
        frame_count = 0
        start_time = time.time()
        last_status_time = start_time
        
        print("✓ Entering main render loop...")
        
        # Main simulation loop
        while not glfw.window_should_close(engine.window):
            frame_start = time.perf_counter()
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Log FPS and status every 2 seconds
            if current_time - last_status_time >= 2.0:
                fps = frame_count / elapsed_time
                cam_pos = camera.getPosition()
                print(f"[{current_time - start_time:.1f}s] FPS: {fps:.1f} | Camera: ({cam_pos.x:.2e}, {cam_pos.y:.2e}, {cam_pos.z:.2e}) | Radius: {camera.radius:.2e}m")
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = current_time
                last_status_time = current_time
            
            # Clear screen
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            
            # Update camera state
            camera.update()
            
            # Generate grid mesh (cached after first generation)
            engine.generate_grid(objects)
            
            # Dispatch compute shader for GPU raytracing
            engine.dispatch_compute(camera)
            
            # Render compute shader output as fullscreen quad (background)
            engine.draw_fullscreen_quad()
            
            # Get cached view-projection matrix (only recalculates if camera moved significantly)
            view_proj_matrix = engine.get_view_proj_matrix(camera)
            
            # Draw the warped spacetime grid on top
            engine.draw_grid(view_proj_matrix)
            
            glfw.swap_buffers(engine.window)
            glfw.poll_events()
    finally:
        print("\n=== Shutting Down ===")
        print("✓ Cleaning up OpenGL resources...")
        engine.shutdown()
        print("✓ Simulation ended successfully")

if __name__ == "__main__":
    main()
