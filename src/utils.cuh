#pragma once

#include <cstddef>

namespace Hyperparams {
    constexpr float AVOID_FACTOR = .005f;
    constexpr float MATCHING_FACTOR = .005f;
    constexpr float CENTERING_FACTOR = .0005f;
    constexpr float BUOY_SPEED = .002f;

    constexpr float  VISION_DISTANCE = .1f;
    constexpr float  AVOID_DISTANCE = .03f;
    constexpr float  MAX_SPEED = .005f;
    constexpr float  MIN_SPEED = .003f;

    constexpr float  TOP_BOUND = .95f;
    constexpr float  RIGHT_BOUND = .95f;
    constexpr float  FAR_BOUND = .95f;
    constexpr float  BOTTOM_BOUND = -.95f;
    constexpr float  LEFT_BOUND = -.95f;
    constexpr float  NEAR_BOUND = -.95f;

    constexpr size_t FLOCK_SIZE = 60000;
};

namespace Universals {
    constexpr unsigned int BOID_VERTICES = 12;
}

namespace DeviceHelpers {
    //helpers
    __device__ inline float3 add(float3 a, float3 b) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ inline float3 sub(float3 a, float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ inline float3 scale(float3 a, float s) {
        return make_float3(a.x * s, a.y * s, a.z * s);
    }

    __device__ inline float dot(float3 a, float3 b) {
        return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    __device__ inline float3 cross(float3 a, float3 b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __device__ inline float3 normalize(float3 a) {
        float invLen = rnorm3df(a.x,a.y,a.z);
        return scale(a, invLen);
    }
}

struct Accumulator {
    float3 pos_avg{0.0f, 0.0f, 0.0f};
    float3 vel_avg{0.0f, 0.0f, 0.0f};
    unsigned int neighboring_boids = 0.0;
    float3 close{0.0f, 0.0f, 0.0f};
};
