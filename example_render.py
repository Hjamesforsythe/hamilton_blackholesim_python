from example_config import *
import example_mesh_factory

class Engine:

    def __init__(self):
        self.init_glfw()
        self.init_gl()

    def init_glfw(self):
        #initialize glfw
        glfw.init()
        #set the window hints
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
             GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
        #this is for forward compatibility especially for mac
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
        #window and share are none
        self.window = glfw.create_window(
            ScreenWidth, ScreenHeight, "Engine Example", None, None)
        #when the window is created, we need to make the context current
        glfw.make_context_current(self.window)
    
    def init_gl(self):
        #clear the color to a green color
        glClearColor(0.05, 0.8, 0.10, 1.0)
        #self.triangle_buffers, self.vao = example_mesh_factory.build_triangle_mesh()
        self.tri_vbo, self.tri_vao = example_mesh_factory.build_triangle_mesh_2()
        self.quad_ebo, self.quad_vbo, self.quad_vao = example_mesh_factory.build_quad_mesh()
        self.shader = create_shader_program("shaders/example_vertex_shader_gpu.txt", "shaders/example_fragment_shader.txt")

    def run(self):
        while not glfw.window_should_close(self.window):

            if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
                glfw.set_window_should_close(self.window, True)
            #detect any events every frame
            glfw.poll_events()

            #double buffer showcase
            glClear(GL_COLOR_BUFFER_BIT)
            #these steps are independant
            glUseProgram(self.shader)
            glBindVertexArray(self.quad_vao)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glfw.swap_buffers(self.window)



    def quit(self):
        #how many vertex arrays to destroy
        #glDeleteBuffers(len(self.triangle_buffers), self.triangle_buffers)
        glDeleteBuffers(2, [self.quad_vbo,self.quad_ebo])
        glDeleteVertexArrays(1, [self.quad_vao])
        glDeleteProgram(self.shader)
        glfw.destroy_window(self.window)
        glfw.terminate()

example_Engine = Engine()
example_Engine.run()
example_Engine.quit()

