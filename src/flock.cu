#include "flock.cuh"
#include <random>
#include <cmath>
#include <cuda/std/cmath>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>

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

__global__ void updateVeloc(Boid* boids, int* grids, int* gridStarts) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Hyperparams::FLOCK_SIZE) {
        Accumulator accum;
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {
                    int neighborGridIdx = grids[idx] + i + j * Hyperparams::X_GRIDS + k * Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS;
                    if(neighborGridIdx < 0 || neighborGridIdx > Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS * Hyperparams::Z_GRIDS)
                        continue;
                    
                    //empty cell
                    if(gridStarts[neighborGridIdx] == -1)
                        continue;
                        
                    for(int neighborIdx = gridStarts[neighborGridIdx];neighborIdx < Hyperparams::FLOCK_SIZE;neighborIdx++)
                    {

                        //if we're out of the grid
                        unsigned int x = cuda::std::floor((boids[neighborIdx].getPosit().x - Hyperparams::LEFT_BOUND) / Hyperparams::VISION_DISTANCE);
                        unsigned int y = cuda::std::floor((boids[neighborIdx].getPosit().y - Hyperparams::BOTTOM_BOUND) / Hyperparams::VISION_DISTANCE);
                        unsigned int z = cuda::std::floor((boids[neighborIdx].getPosit().z - Hyperparams::NEAR_BOUND) / Hyperparams::VISION_DISTANCE);

                        if(neighborGridIdx != x + y * Hyperparams::X_GRIDS + z * Hyperparams::X_GRIDS * Hyperparams::Z_GRIDS)
                            break;

                        //get data for later
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

__global__ void updatePosit(Boid* boids) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        boids[i].step();
        
    }
};

__global__ void genTransform(Boid* boids, float4* transforms) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        cuda::std::array<float4,4> toWrite = DeviceHelpers::transform(boids[i].getPosit(),boids[i].getVeloc());
        memcpy(&transforms[i*4],&toWrite,sizeof(toWrite));
    }
};

__global__ void assignGrid(Boid* boids, int* indices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Hyperparams::FLOCK_SIZE) {
        unsigned int x = cuda::std::floor((boids[i].getPosit().x - Hyperparams::LEFT_BOUND) / Hyperparams::VISION_DISTANCE);
        unsigned int y = cuda::std::floor((boids[i].getPosit().y - Hyperparams::BOTTOM_BOUND) / Hyperparams::VISION_DISTANCE);
        unsigned int z = cuda::std::floor((boids[i].getPosit().z - Hyperparams::NEAR_BOUND) / Hyperparams::VISION_DISTANCE);

        indices[i] = x + y * Hyperparams::X_GRIDS + z * Hyperparams::X_GRIDS * Hyperparams::Z_GRIDS;
    }
};

__global__ void assignGridStarts(int* indices, int* starts) {
    //only use one thread
    //O(N) time but that shouldn't be a problem
    
    //initialize
    for (int i = 0; i < Hyperparams::X_GRIDS * Hyperparams::Y_GRIDS * Hyperparams::Z_GRIDS;i++)
    {
        starts[i] = -1;
    }

    starts[0] = 0;
    for (int i = 0; i < Hyperparams::FLOCK_SIZE; i++) {
        if (i == 0 || indices[i] != indices[i-1]) {
            starts[indices[i]] = i;
        }
    }
};

void Flock::step(float4* transforms) {
    //organize boids into grids
    assignGrid<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids, mpd_gridIndices);
    
    thrust::device_ptr<Boid> d_thrustBoids(mpd_boids);
    thrust::device_ptr<int> d_thrustIndices(mpd_gridIndices);
    thrust::sort_by_key(d_thrustIndices, d_thrustIndices+Hyperparams::FLOCK_SIZE,d_thrustBoids);

    assignGridStarts<<<1,1>>>(mpd_gridIndices, mpd_gridStarts);

    //run boid computations
    updateVeloc<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,mpd_gridIndices,mpd_gridStarts);
    updatePosit<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids);
    genTransform<<<(Hyperparams::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,transforms);
};

Flock::Flock() {
    Boid* boids = (Boid*)malloc(Hyperparams::FLOCK_SIZE*sizeof(Boid));
    for(size_t i = 0; i < Hyperparams::FLOCK_SIZE; i++) {
        boids[i] = randomBoid();
    }

    cudaMalloc(&mpd_boids, Hyperparams::FLOCK_SIZE*sizeof(Boid));
    cudaMemcpy(mpd_boids, boids, Hyperparams::FLOCK_SIZE*sizeof(Boid), cudaMemcpyHostToDevice);
    free(boids);

    cudaMalloc(&mpd_gridIndices, Hyperparams::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, Hyperparams::X_GRIDS*Hyperparams::Y_GRIDS*Hyperparams::Z_GRIDS*sizeof(int));
};

Flock::~Flock() {
    cudaFree(mpd_boids);
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
};
