from example_config import *

def build_triangle_mesh() -> tuple[tuple[int], int]:
    #need float32 for the gpu
    positions = np.array((
        -0.75, -0.75, 0.0,
         0.75, -0.75, 0.0,
         0.0,  0.75, 0.0
    ), dtype = np.float32)

    colors = np.array(
        (0, 1, 2),dtype=np.uint32
    )

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    position_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, position_buffer)
    #upload the data to the GPU
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    #set the attribute pointer
    attribute_index = 0
    #the 3 is because we have 3 floats per vertex
    size = 3
    #the 3*4 is because we have 4 bytes per float
    stride = 12
    offset = 0
    glVertexAttribPointer(attribute_index,size,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)


    color_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    attribute_index = 1
    size = 1
    stride = 4
    offset = 0
    glVertexAttribIPointer(attribute_index,size,GL_UNSIGNED_INT,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)

    return ((position_buffer, color_buffer), vao)

def build_triangle_mesh_2() -> tuple[tuple[int], int]:
    
    vertex_data = np.zeros(3, dtype=data_type_vertex)
    vertex_data[0] = (-0.75, -0.75, 0.0, 0)
    vertex_data[1] = ( 0.75, -0.75, 0.0, 1)
    vertex_data[2] = ( 0.0,  0.75, 0.0, 2)  

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #upload the data to the GPU
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    #set the attribute pointer
    attribute_index = 0
    #the 3 is because we have 3 floats per vertex
    size = 3
    #the 3*4 is because we have 4 bytes per float
    stride = data_type_vertex.itemsize
    offset = 0
    glVertexAttribPointer(attribute_index,size,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)
    offset += 12

    attribute_index = 1
    size = 1
    glVertexAttribIPointer(attribute_index,size,GL_UNSIGNED_INT,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)

    return (vbo, vao)

def build_quad_mesh() -> tuple[int,int, int]:
    
    vertex_data = np.zeros(4, dtype=data_type_vertex)
    vertex_data[0] = (-0.75, -0.75, 0.0, 0)
    vertex_data[1] = ( 0.75, -0.75, 0.0, 1)
    vertex_data[2] = ( 0.75,  0.75, 0.0, 2)  
    vertex_data[3] = ( -0.75,  0.75, 0.0, 1)

    index_data = np.array((0,1,2,2,3,0), dtype=np.ubyte)
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    #upload the data to the GPU
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    #set the attribute pointer
    attribute_index = 0
    #the 3 is because we have 3 floats per vertex
    size = 3
    #the 3*4 is because we have 4 bytes per float
    stride = data_type_vertex.itemsize
    offset = 0
    glVertexAttribPointer(attribute_index,size,GL_FLOAT,GL_FALSE,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)
    offset += 12

    attribute_index = 1
    size = 1
    glVertexAttribIPointer(attribute_index,size,GL_UNSIGNED_INT,stride,ctypes.c_void_p(offset))
    glEnableVertexAttribArray(attribute_index)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    #upload the data to the GPU
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes, index_data, GL_STATIC_DRAW)

    return (ebo, vbo, vao)