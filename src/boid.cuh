#pragma once

#include "utils.cuh"

class Boid {
    private:
        float3 m_posit, m_veloc, m_newVeloc;
    public: 
        // still need to initialize
        __host__ __device__ Boid() {}

        Boid(const float3& p, const float3& v) :
        m_posit(p),
        m_veloc(v),
        m_newVeloc(v) {};
        __device__ float3 getPosit() const { return m_posit; };
        __device__ float3 getVeloc() const { return m_veloc; };
        __device__ void setNewVeloc(const float3 v) {m_newVeloc = v; };
        __device__ void step() {
            m_veloc.x = m_newVeloc.x; 
            m_veloc.y = m_newVeloc.y;
            m_veloc.z = m_newVeloc.z;
            m_posit.x += m_veloc.x;
            m_posit.y += m_veloc.y;
            m_posit.z += m_veloc.z; 

            //keep bounded, just teleport to other side
            m_posit.x = m_posit.x > Hyperparams::RIGHT_BOUND ? m_posit.x - (Hyperparams::RIGHT_BOUND - Hyperparams::LEFT_BOUND) : m_posit.x;
            m_posit.x = m_posit.x < Hyperparams::LEFT_BOUND ? m_posit.x + (Hyperparams::RIGHT_BOUND - Hyperparams::LEFT_BOUND) : m_posit.x;
            m_posit.y = m_posit.y > Hyperparams::TOP_BOUND ? m_posit.y - (Hyperparams::TOP_BOUND - Hyperparams::BOTTOM_BOUND) : m_posit.y;
            m_posit.y = m_posit.y < Hyperparams::BOTTOM_BOUND ? m_posit.y + (Hyperparams::TOP_BOUND - Hyperparams::BOTTOM_BOUND) : m_posit.y;
            m_posit.z = m_posit.z > Hyperparams::FAR_BOUND ? m_posit.z - (Hyperparams::FAR_BOUND - Hyperparams::NEAR_BOUND) : m_posit.z;
            m_posit.z = m_posit.z < Hyperparams::NEAR_BOUND ? m_posit.z + (Hyperparams::FAR_BOUND - Hyperparams::NEAR_BOUND) : m_posit.z;
        };
};
