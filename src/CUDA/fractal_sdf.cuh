#pragma once

#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"

namespace sdf{
    enum{
        MANDELBULB
    };
    // Computes Signed Distance Function to the Mandelbulb fractal (degree 8)
    // Based off work by Inigo Quilez: https://iquilezles.org/articles/mandelbulb/
    __device__ float mandelbulb(const Cuda::vec3 pos, int maxIters){ 
        Cuda::vec3 w = pos;
        float dz = 1.0;
        float m;

        for(int i = 0; i < maxIters; i++){
            m = dot(w, w);
            if(m > 2.0) break;
            float invr = 1/sqrtf(m);
            float r8 = m * m * m * m;
            dz = (8.0 * r8 * invr * dz ) + 1.0;

            float b = 8.0 * acosf(w.y * invr);
            float a = 8.0 * atan2f(w.x, w.z);
            float sinfb = sinf(b);
            w = pos + r8 * Cuda::vec3(sinfb * sinf(a), cosf(b), sinfb * cosf(a));
        }
        float d = 0.25 * log(m) * sqrt(m) / dz;

        return d;
    }
}
