#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <iostream>

#include "flock.cuh"

const char* vertSrc = "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in mat4 instance;\n"
    "void main() { gl_Position = instance * vec4(aPos, 1.0); }\n";

const char* fragSrc = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "void main() { FragColor = vec4(1.0, 0.5, 0.2, 1.0); }\n";

// -- settings --
const int    WINDOW_WIDTH  = 1280;
const int    WINDOW_HEIGHT = 720;

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
        WINDOW_WIDTH, WINDOW_HEIGHT, "Boids", nullptr, nullptr
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

    // boid structure
    float boidVerts[] = {
        0.0f, 0.005f, 0.0f,
        -0.002598f, 0.0f, -0.0015f,
        0.0f, 0.0f, 0.003f,

        0.0f, 0.005f, 0.0f,
        0.002598f, 0.0f, -0.0015f,
        -0.002598f, 0.0f, -0.0015f,

        0.0f, 0.005f, 0.0f,
        0.0f, 0.0f, 0.003f,
        0.002598f, 0.0f, -0.0015f,

        0.0f, 0.0f, 0.003f,
        0.002598f, 0.0f, -0.0015f,
        -0.002598f, 0.0f, -0.0015f,
    };

    // Allocate vertex data
    unsigned int boidVAO, boidVBO;
    glGenVertexArrays(1, &boidVAO);
    glGenBuffers(1, &boidVBO);

    glBindVertexArray(boidVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boidVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3)*12, boidVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // Allocate instance data
    unsigned int instVBO;
    glGenBuffers(1, &instVBO);

    glBindBuffer(GL_ARRAY_BUFFER, instVBO);
    size_t matSize = 4*sizeof(float4);
    glBufferData(GL_ARRAY_BUFFER, matSize*Hyperparams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, matSize , nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, matSize , (void*)(1*sizeof(float4)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, matSize , (void*)(2*sizeof(float4)));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, matSize , (void*)(3*sizeof(float4)));
    glEnableVertexAttribArray(4);

    glVertexAttribDivisor(1,1);
    glVertexAttribDivisor(2,1);
    glVertexAttribDivisor(3,1);
    glVertexAttribDivisor(4,1);

    //unbind for now
    glBindVertexArray(0);
    
    // create boid data
    Flock flock;
    
    // register with cuda
    cudaGraphicsResource* cudaVBO;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaVBO, instVBO, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess)
        std::cout << cudaGetErrorString(err) << std::endl;

    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        // run calculations
        size_t size;
        float4* transforms;
        
        cudaGraphicsMapResources(1,&cudaVBO,0);
        cudaGraphicsResourceGetMappedPointer((void**)&transforms,&size,cudaVBO);
        
        flock.step(transforms);
        
        cudaGraphicsUnmapResources(1, &cudaVBO, 0);
        
        // draw
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glBindVertexArray(boidVAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, 12, Hyperparams::FLOCK_SIZE);

        glfwSwapBuffers(window);
        glfwPollEvents();

        //fps
        double currentTime = glfwGetTime();
        frameCount++;
        if (currentTime - lastTime >= 1.0) {
            double fps = frameCount / (currentTime - lastTime);
            char title[64];
            snprintf(title,64,"Boids | FPS: %.1f", fps);
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            lastTime = currentTime;
        }
    }

    
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
