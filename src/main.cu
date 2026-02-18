#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "flock.cuh"

const char* vertSrc = "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "void main() { gl_Position = vec4(aPos, 1.0); }\n";

const char* fragSrc = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main() { FragColor = vec4(1.0, 0.5, 0.2, 1.0); }\n";

// -- settings --
const int    WINDOW_WIDTH  = 1280;
const int    WINDOW_HEIGHT = 720;
const char*  WINDOW_TITLE  = "Boids";

void onKeyPress(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main() {

    // Set up window
    if (!glfwInit()) {
        std::cerr << "Failed to initialise GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    GLFWwindow* window = glfwCreateWindow(
        WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr
    );

    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, onKeyPress);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    

    // Compile shaders
    unsigned int vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertSrc, NULL);
    glCompileShader(vert);

    unsigned int frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragSrc, NULL);
    glCompileShader(frag);

    unsigned int program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glDeleteShader(vert);
    glDeleteShader(frag);

    // Allocate Vertex Data
    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*Hyperparams::FLOCK_SIZE*Universals::BOID_VERTICES, NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    //unbind for now
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // create boid data
    Flock flock;
    
    // register with cuda
    cudaGraphicsResource* cudaVBO;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;
    

    while (!glfwWindowShouldClose(window)) {
        // run calculations
        size_t size;
        float3* verts;
        cudaGraphicsMapResources(1,&cudaVBO,0);
        cudaGraphicsResourceGetMappedPointer((void**)&verts,&size,cudaVBO);

        flock.step(verts);

        cudaGraphicsUnmapResources(1, &cudaVBO, 0);

        // draw
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, Hyperparams::FLOCK_SIZE*Universals::BOID_VERTICES);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
