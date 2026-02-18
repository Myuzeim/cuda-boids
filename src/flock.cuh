#pragma once

#include "boid.cuh"
#include "utils.cuh"

//input: a list of n boids.
__global__ void updateVeloc(Boid* boids, const size_t n);
__global__ void updatePosit(Boid* boids, const size_t n);
__global__ void genTransform(Boid* boids, float3* verts, const unsigned int n);

// a flock of boids
// set # of boids in Hyperparams.FLOCK_SIZE
class Flock {
    private:
        Boid* mpd_boids;
        Boid randomBoid();
    
    public:
        void step(float4* transforms);
        Flock();
        ~Flock();

};
