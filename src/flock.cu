#include "flock.cuh"
#include <random>
#include <cmath>

Boid Flock::randomBoid() {
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float len = std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (Hyperparams::MIN_SPEED + Hyperparams::MAX_SPEED) / 2;
    float scale = magnitude / len;

    float3 veloc(vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale);

    std::uniform_real_distribution<float> distx(Hyperparams::LEFT_BOUND,Hyperparams::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(Hyperparams::BOTTOM_BOUND,Hyperparams::TOP_BOUND);
    std::uniform_real_distribution<float> distz(Hyperparams::NEAR_BOUND,Hyperparams::FAR_BOUND);

    float3 posit(distx(rng),disty(rng),distz(rng));

    return Boid(posit, veloc);
};

__global__ void updateVeloc(Boid* boids, const size_t n) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n) {
        Accumulator accum;
        for(int neighborIdx = 0; neighborIdx < Hyperparams::FLOCK_SIZE; neighborIdx++) {
            
            float3 d = DeviceHelpers::sub(boids[idx].getPosit(), boids[neighborIdx].getPosit());
            float sqDist = d.x*d.x + d.y*d.y + d.z*d.z;
            
            if (sqDist < Hyperparams::AVOID_DISTANCE*Hyperparams::AVOID_DISTANCE) {
                //Avoiding
                accum.close = DeviceHelpers::add(accum.close, DeviceHelpers::sub(boids[idx].getPosit(),boids[neighborIdx].getPosit()));
            } else if (sqDist < Hyperparams::VISION_DISTANCE*Hyperparams::VISION_DISTANCE) {
                // Centering/Matching
                accum.pos_avg = DeviceHelpers::add(accum.pos_avg, boids[neighborIdx].getPosit());
                accum.vel_avg = DeviceHelpers::add(accum.vel_avg, boids[neighborIdx].getVeloc());
                accum.neighboring_boids += 1;
            }
        }
        
        float3 newVeloc = boids[idx].getVeloc();

        if (accum.neighboring_boids > 0) {
            //add centering/matching
            accum.pos_avg = DeviceHelpers::scale(accum.pos_avg,1.0f/((float) accum.neighboring_boids));
            accum.vel_avg = DeviceHelpers::scale(accum.vel_avg,1.0f/((float) accum.neighboring_boids));

            newVeloc.x = newVeloc.x + 
                (accum.pos_avg.x - boids[idx].getPosit().x) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.x - boids[idx].getVeloc().x) * Hyperparams::MATCHING_FACTOR;
            newVeloc.y = newVeloc.y + 
                (accum.pos_avg.y - boids[idx].getPosit().y) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.y - boids[idx].getVeloc().y) * Hyperparams::MATCHING_FACTOR;
            newVeloc.z = newVeloc.z + 
                (accum.pos_avg.z - boids[idx].getPosit().z) * Hyperparams::CENTERING_FACTOR +
                (accum.vel_avg.z - boids[idx].getVeloc().z) * Hyperparams::MATCHING_FACTOR;
        }

        // add avoiding
        newVeloc.x = newVeloc.x + accum.close.x*Hyperparams::AVOID_FACTOR;
        newVeloc.y = newVeloc.y + accum.close.y*Hyperparams::AVOID_FACTOR;
        newVeloc.z = newVeloc.z + accum.close.z*Hyperparams::AVOID_FACTOR;

        // buoy (bounce off sides)
        if (boids[idx].getPosit().x < Hyperparams::LEFT_BOUND)
            newVeloc.x += Hyperparams::BUOY_SPEED;
        if (boids[idx].getPosit().x > Hyperparams::RIGHT_BOUND)
            newVeloc.x -= Hyperparams::BUOY_SPEED;
        if (boids[idx].getPosit().y < Hyperparams::BOTTOM_BOUND)
            newVeloc.y += Hyperparams::BUOY_SPEED;
        if (boids[idx].getPosit().y > Hyperparams::TOP_BOUND)
            newVeloc.y -= Hyperparams::BUOY_SPEED;
        if (boids[idx].getPosit().z < Hyperparams::FAR_BOUND)
            newVeloc.z += Hyperparams::BUOY_SPEED;
        if (boids[idx].getPosit().z > Hyperparams::NEAR_BOUND)
            newVeloc.z -= Hyperparams::BUOY_SPEED;

        float invSpeed = rnorm3df(newVeloc.x, newVeloc.y, newVeloc.z);
        if ((1.0f / invSpeed) < Hyperparams::MIN_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Hyperparams::MIN_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Hyperparams::MIN_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Hyperparams::MIN_SPEED;
        }
        if ((1.0f / invSpeed) > Hyperparams::MAX_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Hyperparams::MAX_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Hyperparams::MAX_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Hyperparams::MAX_SPEED;
        }

        boids[idx].setNewVeloc(newVeloc);
    }
};

__global__ void updatePosit(Boid* boids, const size_t n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        boids[i].step();
    }
};

__global__ void genVert(Boid* boids, float3* verts, const unsigned int n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        float3 velNormal = DeviceHelpers::normalize(boids[i].getVeloc());
        float3 cross1, cross2;
        if (velNormal.y < .99) {
            cross1 = DeviceHelpers::cross(velNormal, {0.0f,1.0f,0.0f});
            cross2 = DeviceHelpers::cross(velNormal, cross1);
        } else {
            cross1 = DeviceHelpers::cross(velNormal, {1.0f,0.0f,0.0f});
            cross2 = DeviceHelpers::cross(velNormal, cross1);
        }
        cross1 = DeviceHelpers::scale(cross1,.005f);
        cross2 = DeviceHelpers::scale(cross2,.005f);
        float3 velSmall = DeviceHelpers::scale(velNormal,.005f);
        unsigned int vertIndex = i * Universals::BOID_VERTICES;
        verts[vertIndex] = DeviceHelpers::add(boids[i].getPosit(),velSmall);
        verts[vertIndex+1] = DeviceHelpers::add(boids[i].getPosit(),cross1);
        verts[vertIndex+2] = DeviceHelpers::add(boids[i].getPosit(),cross2);

        verts[vertIndex+3] = DeviceHelpers::add(boids[i].getPosit(),velSmall);
        verts[vertIndex+4] = DeviceHelpers::add(boids[i].getPosit(),cross2);
        verts[vertIndex+5] = DeviceHelpers::sub(boids[i].getPosit(),cross1);

        verts[vertIndex+6] = DeviceHelpers::add(boids[i].getPosit(),velSmall);
        verts[vertIndex+7] = DeviceHelpers::sub(boids[i].getPosit(),cross1);
        verts[vertIndex+8] = DeviceHelpers::sub(boids[i].getPosit(),cross2);

        verts[vertIndex+9] = DeviceHelpers::add(boids[i].getPosit(), velSmall);
        verts[vertIndex+10] = DeviceHelpers::sub(boids[i].getPosit(), cross2);
        verts[vertIndex+11] = DeviceHelpers::add(boids[i].getPosit(), cross1);
    }
};

void Flock::step(float3* verts) {
    updateVeloc<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,Hyperparams::FLOCK_SIZE);
    updatePosit<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,Hyperparams::FLOCK_SIZE);
    genVert<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,verts,Hyperparams::FLOCK_SIZE);
};

Flock::Flock() {
    Boid* boids = (Boid*)malloc(Hyperparams::FLOCK_SIZE*sizeof(Boid));
    for(size_t i = 0; i < Hyperparams::FLOCK_SIZE; i++) {
        boids[i] = randomBoid();
    }

    cudaMalloc(&mpd_boids, Hyperparams::FLOCK_SIZE*sizeof(Boid));
    cudaMemcpy(mpd_boids, boids, Hyperparams::FLOCK_SIZE*sizeof(Boid), cudaMemcpyHostToDevice);
    free(boids);
};

Flock::~Flock() {
    cudaFree(mpd_boids);
};
