#include "flock.cuh"
#include <random>
#include <cmath>
#include <cuda/std/cmath>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>

Boid Flock::randomBoid() {
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    float vxUnscaled = dist(rng);
    float vyUnscaled = dist(rng);
    float vzUnscaled = dist(rng);

    float len = std::sqrt(vxUnscaled*vxUnscaled + vyUnscaled*vyUnscaled + vzUnscaled*vzUnscaled);
    float magnitude = (Params::MIN_SPEED + Params::MAX_SPEED) / 2;
    float scale = magnitude / len;

    float3 veloc(vxUnscaled*scale,vyUnscaled*scale,vzUnscaled*scale);

    std::uniform_real_distribution<float> distx(Params::LEFT_BOUND,Params::RIGHT_BOUND);
    std::uniform_real_distribution<float> disty(Params::BOTTOM_BOUND,Params::TOP_BOUND);
    std::uniform_real_distribution<float> distz(Params::NEAR_BOUND,Params::FAR_BOUND);

    float3 posit(distx(rng),disty(rng),distz(rng));

    return Boid(posit, veloc);
};

__global__ void updateVeloc(Boid* boids, int* grids, int* gridStarts, int* boidIndices) {
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < Params::FLOCK_SIZE) {
        Accumulator accum;

        int boidIdx = boidIndices[idx];
        
        //surrounding 9 grids
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++)
                {
                    int neighborGridIdx = grids[idx] + i + j * Params::X_GRIDS + k * Params::X_GRIDS * Params::Y_GRIDS;
                    if(neighborGridIdx < 0 || neighborGridIdx >= Params::X_GRIDS * Params::Y_GRIDS * Params::Z_GRIDS)
                        continue;

                    //empty cell
                    if(gridStarts[neighborGridIdx] >= Params::FLOCK_SIZE)
                        continue;
                        
                    for(int neighborIdx = gridStarts[neighborGridIdx];neighborIdx < Params::FLOCK_SIZE;neighborIdx++)
                    {
                        //if we're out of the grid
                        if(grids[neighborIdx] != neighborGridIdx)
                            break;

                        //get data for later
                        float3 d = DeviceHelpers::sub(boids[boidIdx].getPosit(), boids[boidIndices[neighborIdx]].getPosit());
                        float sqDist = d.x*d.x + d.y*d.y + d.z*d.z;

                        if (sqDist < Params::AVOID_DISTANCE*Params::AVOID_DISTANCE) {
                            //Avoiding
                            accum.close = DeviceHelpers::add(accum.close, DeviceHelpers::sub(boids[boidIdx].getPosit(),boids[boidIndices[neighborIdx]].getPosit()));
                        } else if (sqDist < Params::VISION_DISTANCE*Params::VISION_DISTANCE) {
                            // Centering/Matching
                            accum.pos_avg = DeviceHelpers::add(accum.pos_avg, boids[boidIndices[neighborIdx]].getPosit());
                            accum.vel_avg = DeviceHelpers::add(accum.vel_avg, boids[boidIndices[neighborIdx]].getVeloc());
                            accum.neighboring_boids += 1;
                        }
                    }
                }
        
        float3 newVeloc = boids[boidIdx].getVeloc();

        if (accum.neighboring_boids > 0) {
            //add centering/matching
            accum.pos_avg = DeviceHelpers::scale(accum.pos_avg,1.0f/((float) accum.neighboring_boids));
            accum.vel_avg = DeviceHelpers::scale(accum.vel_avg,1.0f/((float) accum.neighboring_boids));

            newVeloc.x = newVeloc.x + 
                (accum.pos_avg.x - boids[boidIdx].getPosit().x) * Params::CENTERING_FACTOR +
                (accum.vel_avg.x - boids[boidIdx].getVeloc().x) * Params::MATCHING_FACTOR;
            newVeloc.y = newVeloc.y + 
                (accum.pos_avg.y - boids[boidIdx].getPosit().y) * Params::CENTERING_FACTOR +
                (accum.vel_avg.y - boids[boidIdx].getVeloc().y) * Params::MATCHING_FACTOR;
            newVeloc.z = newVeloc.z + 
                (accum.pos_avg.z - boids[boidIdx].getPosit().z) * Params::CENTERING_FACTOR +
                (accum.vel_avg.z - boids[boidIdx].getVeloc().z) * Params::MATCHING_FACTOR;
        }

        // add avoiding
        newVeloc.x = newVeloc.x + accum.close.x*Params::AVOID_FACTOR;
        newVeloc.y = newVeloc.y + accum.close.y*Params::AVOID_FACTOR;
        newVeloc.z = newVeloc.z + accum.close.z*Params::AVOID_FACTOR;

        float invSpeed = rnorm3df(newVeloc.x, newVeloc.y, newVeloc.z);
        if ((1.0f / invSpeed) < Params::MIN_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Params::MIN_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Params::MIN_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Params::MIN_SPEED;
        }
        if ((1.0f / invSpeed) > Params::MAX_SPEED) {
            newVeloc.x = newVeloc.x * invSpeed * Params::MAX_SPEED;
            newVeloc.y = newVeloc.y * invSpeed * Params::MAX_SPEED;
            newVeloc.z = newVeloc.z * invSpeed * Params::MAX_SPEED;
        }

        boids[boidIdx].setNewVeloc(newVeloc);
    }
};

__global__ void updatePosit(Boid* boids) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        boids[i].step();
        
    }
};

__global__ void genTransform(Boid* boids, float4* transforms) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        cuda::std::array<float4,4> toWrite = DeviceHelpers::transform(boids[i].getPosit(),boids[i].getVeloc());
        memcpy(&transforms[i*4],&toWrite,sizeof(toWrite));
    }
};

__global__ void assignGrid(Boid* boids, int* gridIndices) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < Params::FLOCK_SIZE) {
        unsigned int x = cuda::std::floor((boids[i].getPosit().x - Params::LEFT_BOUND) / Params::VISION_DISTANCE);
        unsigned int y = cuda::std::floor((boids[i].getPosit().y - Params::BOTTOM_BOUND) / Params::VISION_DISTANCE);
        unsigned int z = cuda::std::floor((boids[i].getPosit().z - Params::NEAR_BOUND) / Params::VISION_DISTANCE);

        gridIndices[i] = x + y * Params::X_GRIDS + z * Params::X_GRIDS * Params::Z_GRIDS;
    }
};

void Flock::step(float4* transforms) {
    
    //auto start = std::chrono::high_resolution_clock::now()//;

    //organize boids into grids
    assignGrid<<<(Params::FLOCK_SIZE + 255)/256,256>>>(mpd_boids, mpd_gridIndices);

    thrust::device_ptr<int> d_thrustBoidIndices(mpd_boidIndices);
    thrust::device_ptr<int> d_thrustGridIndices(mpd_gridIndices);
    thrust::sequence(d_thrustBoidIndices, d_thrustBoidIndices + Params::FLOCK_SIZE);
    thrust::sort_by_key(d_thrustGridIndices, d_thrustGridIndices+Params::FLOCK_SIZE,d_thrustBoidIndices);

    //find starting point of each grid
    thrust::device_ptr<int> d_thrustGridStarts(mpd_gridStarts);
    thrust::lower_bound(d_thrustGridIndices, d_thrustGridIndices + Params::FLOCK_SIZE,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS),
                    d_thrustGridStarts);


    //run boid computations
    updateVeloc<<<(Params::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,mpd_gridIndices,mpd_gridStarts,mpd_boidIndices);
    updatePosit<<<(Params::FLOCK_SIZE + 255)/256,256>>>(mpd_boids);
    genTransform<<<(Params::FLOCK_SIZE + 255)/256,256>>>(mpd_boids,transforms);
};

Flock::Flock() {
    Boid* boids = (Boid*)malloc(Params::FLOCK_SIZE*sizeof(Boid));
    for(size_t i = 0; i < Params::FLOCK_SIZE; i++) {
        boids[i] = randomBoid();
    }

    cudaMalloc(&mpd_boids, Params::FLOCK_SIZE*sizeof(Boid));
    cudaMemcpy(mpd_boids, boids, Params::FLOCK_SIZE*sizeof(Boid), cudaMemcpyHostToDevice);
    free(boids);

    cudaMalloc(&mpd_gridIndices, Params::FLOCK_SIZE*sizeof(int));

    cudaMalloc(&mpd_gridStarts, Params::X_GRIDS*Params::Y_GRIDS*Params::Z_GRIDS*sizeof(int));

    cudaMalloc(&mpd_boidIndices, Params::FLOCK_SIZE*sizeof(int));
};

Flock::~Flock() {
    cudaFree(mpd_boids);
    cudaFree(mpd_gridIndices);
    cudaFree(mpd_gridStarts);
    cudaFree(mpd_boidIndices);
};
