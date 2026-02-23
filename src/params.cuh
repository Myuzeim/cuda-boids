#pragma once

#include <cstddef>
#include <cuda_fp16.h>

namespace HostParams {
        constexpr float AVOID_FACTOR = 2.f;
        constexpr float MATCHING_FACTOR = .6f;
        constexpr float CENTERING_FACTOR = .15f;

        constexpr float  VISION_DISTANCE = 2.f;
        constexpr float  AVOID_DISTANCE = .5f;
        constexpr float  MAX_SPEED = .7f;
        constexpr float  MIN_SPEED = .3f;

        constexpr float  TOP_BOUND = 150.f;
        constexpr float  RIGHT_BOUND = 150.f;
        constexpr float  FAR_BOUND = 150.f;
        constexpr float  BOTTOM_BOUND = -150.f;
        constexpr float  LEFT_BOUND = -150.f;
        constexpr float  NEAR_BOUND = -150.f;


        constexpr int FLOCK_SIZE = 5000000;
        constexpr int BLOCK_SIZE = 256;
    
        constexpr unsigned int uint_ceil(const float f)
        {
            unsigned int i = static_cast<int>(f);
            float ff = static_cast<float>(f);
            return ff > i ? i + 1 : i;
        }


        constexpr unsigned int X_GRIDS = uint_ceil((HostParams::RIGHT_BOUND - HostParams::LEFT_BOUND) / HostParams::VISION_DISTANCE);
        constexpr unsigned int Y_GRIDS = uint_ceil((HostParams::TOP_BOUND - HostParams::BOTTOM_BOUND) / HostParams::VISION_DISTANCE);
        constexpr unsigned int Z_GRIDS = uint_ceil((HostParams::FAR_BOUND - HostParams::NEAR_BOUND) / HostParams::VISION_DISTANCE);
        constexpr unsigned int AREA_GRIDS = X_GRIDS * Y_GRIDS * Z_GRIDS;

        constexpr float WORLD_WIDTH = HostParams::RIGHT_BOUND - HostParams::LEFT_BOUND;
        constexpr float WORLD_HEIGHT = HostParams::TOP_BOUND - HostParams::BOTTOM_BOUND;
        constexpr float WORLD_DEPTH = HostParams::FAR_BOUND - HostParams::NEAR_BOUND;
};