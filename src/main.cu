#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <iostream>

#include "flock.cuh"

const char* vertSrc = "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;\n"
    "layout(location = 1) in float x;\n"
    "layout(location = 2) in float vx;\n"
    "layout(location = 3) in float y;\n"
    "layout(location = 4) in float vy;\n"
    "layout(location = 5) in float z;\n"
    "layout(location = 6) in float vz;\n"
    "uniform mat4 viewProj;\n"
    "void main() { \n"
    "   vec3 pos = vec3(x, y, z);\n"
    "   vec3 up  = normalize(vec3(vx, vy, vz));\n"
    "   vec3 ref   = abs(up.y) > 0.9999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);\n"
    "   vec3 right = normalize(cross(ref, up));\n"
    "   vec3 fwd   = cross(right, up);\n"
    "   \n"
    "   vec3 worldPos = right * (aPos.x * 50.0)\n"
    "                 + up    * (aPos.y * 50.0)"
    "                 + fwd   * (aPos.z * 50.0)\n"
    "                 + pos;\n"
    "   \n"
    "   gl_Position = viewProj * vec4(worldPos, 1.0);\n"
    "}\n";

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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " — " << cudaGetErrorString(err) << "\n"; \
            return -1; \
        } \
    } while(0)

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

    // Set up camera
    struct CamState {
        float yaw = 0.f, pitch = 0.0f, dist = 500.f;
        double lastX = 0, lastY = 0;
        bool dragging = false;
    };
    CamState cam;
    glfwSetWindowUserPointer(window, &cam);

    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int btn, int action, int) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        if (btn == GLFW_MOUSE_BUTTON_LEFT) {
            c.dragging = (action == GLFW_PRESS);
            glfwGetCursorPos(w, &c.lastX, &c.lastY);
        }
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        if (!c.dragging) return;
        c.yaw   += (float)(x - c.lastX) * 0.005f;
        c.pitch += (float)(y - c.lastY) * 0.005f;
        c.pitch  = std::clamp(c.pitch, -1.5f, 1.5f); // don't flip over the poles
        c.lastX = x; c.lastY = y;
    });

    glfwSetScrollCallback(window, [](GLFWwindow* w, double, double dy) {
        auto& c = *(CamState*)glfwGetWindowUserPointer(w);
        c.dist *= (float)std::pow(0.9, dy); // scroll up = zoom in
        c.dist  = std::clamp(c.dist, 0.5f, 700.f);
    });

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

    int uView = glGetUniformLocation(program, "view");
    int uProj = glGetUniformLocation(program, "proj");

    // boid structure
    half boidVerts[] = {
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
    glBufferData(GL_ARRAY_BUFFER, sizeof(half)*3*3, boidVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_HALF_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    // Allocate instance data
    size_t vecSize = sizeof(__half2);
    
    unsigned int xvxVBO[2];
    glGenBuffers(2, xvxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, xvxVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, xvxVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(1, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1,1);

    glVertexAttribPointer(2, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , (void*)sizeof(half));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2,1);

    unsigned int yvyVBO[2];
    glGenBuffers(2, yvyVBO);
    glBindBuffer(GL_ARRAY_BUFFER, yvyVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, yvyVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(3, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , nullptr);
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3,1);

    glVertexAttribPointer(4, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , (void*)sizeof(half));
    glEnableVertexAttribArray(4);
    glVertexAttribDivisor(4,1);

    unsigned int zvzVBO[2];
    glGenBuffers(2, zvzVBO);
    glBindBuffer(GL_ARRAY_BUFFER, zvzVBO[0]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, zvzVBO[1]);
    glBufferData(GL_ARRAY_BUFFER, vecSize*HostParams::FLOCK_SIZE,nullptr,GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(5, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , nullptr);
    glEnableVertexAttribArray(5);
    glVertexAttribDivisor(5,1);

    glVertexAttribPointer(6, 1, GL_HALF_FLOAT, GL_FALSE, vecSize , (void*)sizeof(half));
    glEnableVertexAttribArray(6);
    glVertexAttribDivisor(6,1);

    //unbind for now
    glBindVertexArray(0);
    
    // create boid data
    Flock flock;
    
    // register with cuda
    cudaGraphicsResource* cudaXvxVBO[2];
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaXvxVBO[0], xvxVBO[0], cudaGraphicsMapFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaXvxVBO[1], xvxVBO[1], cudaGraphicsMapFlagsNone));
    
    cudaGraphicsResource* cudaYvyVBO[2];
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaYvyVBO[0], yvyVBO[0], cudaGraphicsMapFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaYvyVBO[1], yvyVBO[1], cudaGraphicsMapFlagsNone));

    cudaGraphicsResource* cudaZvzVBO[2];
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaZvzVBO[0], zvzVBO[0], cudaGraphicsMapFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaZvzVBO[1], zvzVBO[1], cudaGraphicsMapFlagsNone));

    int frontBuffer = 0;
    int backBuffer = 1;
    size_t size;
    __half2* xvxs;
    __half2* yvys;
    __half2* zvzs;
    __half2* xvxs2;
    __half2* yvys2;
    __half2* zvzs2;

    CUDA_CHECK(cudaGraphicsMapResources(1,&cudaXvxVBO[frontBuffer],0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&xvxs,&size,cudaXvxVBO[frontBuffer]));
    
    CUDA_CHECK(cudaGraphicsMapResources(1,&cudaYvyVBO[frontBuffer],0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&yvys,&size,cudaYvyVBO[frontBuffer]));

    CUDA_CHECK(cudaGraphicsMapResources(1,&cudaZvzVBO[frontBuffer],0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&zvzs,&size,cudaZvzVBO[frontBuffer]));
    
    flock.genRand(xvxs, yvys, zvzs);
    
    CUDA_CHECK(cudaGraphicsUnmapResources(1, cudaXvxVBO, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, cudaYvyVBO, 0));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, cudaZvzVBO, 0));


    double lastTime = glfwGetTime();
    int frameCount = 0;

    while (!glfwWindowShouldClose(window)) {
        // run calculations
        
        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaXvxVBO[frontBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&xvxs,&size,cudaXvxVBO[frontBuffer]));
        
        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaYvyVBO[frontBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&yvys,&size,cudaYvyVBO[frontBuffer]));

        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaZvzVBO[frontBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&zvzs,&size,cudaZvzVBO[frontBuffer]));

        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaXvxVBO[backBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&xvxs2,&size,cudaXvxVBO[backBuffer]));
        
        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaYvyVBO[backBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&yvys2,&size,cudaYvyVBO[backBuffer]));

        CUDA_CHECK(cudaGraphicsMapResources(1,&cudaZvzVBO[backBuffer],0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&zvzs2,&size,cudaZvzVBO[backBuffer]));
        
        flock.step(xvxs, yvys, zvzs, xvxs2, yvys2, zvzs2);
        
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaXvxVBO[frontBuffer], 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaYvyVBO[frontBuffer], 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaZvzVBO[frontBuffer], 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaXvxVBO[backBuffer], 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaYvyVBO[backBuffer], 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaZvzVBO[backBuffer], 0));
        
        // Move camera
        glm::vec3 eye(
            cam.dist * cosf(cam.pitch) * sinf(cam.yaw),
            cam.dist * sinf(cam.pitch),
            cam.dist * cosf(cam.pitch) * cosf(cam.yaw)
        );
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspective(
            glm::radians(45.f),
            (float)WINDOW_WIDTH / WINDOW_HEIGHT,
            0.01f, 1000.f
        );
        glm::mat4 viewProj = proj * view;

        glUseProgram(program);
        int uViewProj = glGetUniformLocation(program, "viewProj");
        glUniformMatrix4fv(uViewProj, 1, GL_FALSE, glm::value_ptr(viewProj));
        
        // draw
        glClear(GL_COLOR_BUFFER_BIT);
        glBindVertexArray(boidVAO);
        //glDrawArraysInstanced(GL_TRIANGLES, 0, 12, HostParams::FLOCK_SIZE);

        glfwSwapBuffers(window);

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

        //std::swap(frontBuffer, backBuffer);

        glfwPollEvents();
    }

    
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
