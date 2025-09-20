import glfw 
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from utilities import ShaderUtils, BufferUtils

ScreenWidth = 800
ScreenHeight = 600

#Definitions: 
#Vertex
#A vertex is a set of all attributes needed to draw a pixel
#it may contain position, color, texture coordinates, etc.

#Rasterizer
#interpolates between vertices, works out which points to draw and what their attributes are

#Fragment
#determines the final color  or attribute of each individual pixel

#Frame Buffer
#stores the final image, pixels are drawn to the frame buffer

#Vertex Buffer
#a memory allocation on the GPU, ones and zeros but with no meaning attached

#Attribute Pointer
#describes how the GPU can interpret a vertex buffer and extract individual attributes for the soup

#Vertex Array
#A convienince object which remembers buffers and attributes

# Use centralized vertex data type
data_type_vertex = BufferUtils.create_vertex_data_type()

# Use centralized shader utilities
def create_shader_program(vertex_filepath: str, fragment_filepath: str) -> int:
    return ShaderUtils.create_shader_program(vertex_filepath, fragment_filepath)

def create_shader_module(filepath: str, module_type: int) -> int:
    return ShaderUtils.create_shader_module(filepath, module_type)

