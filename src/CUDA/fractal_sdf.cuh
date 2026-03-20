#pragma once

#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"

using namespace Cuda;

namespace sdf{
    enum{
        MANDELBULB
    };
    // Computes Signed Distance Function to the Mandelbulb fractal (degree 8)
    // Based off work by Inigo Quilez: https://iquilezles.org/articles/mandelbulb/
    __device__ float mandelbulb(vec3 pos, int maxIters){ 
        vec3 w = pos;
        float m = dot(w, w);
        float dz = 1.0;

        for(int i = 0; i < maxIters; i++){
            dz = (8.0 * pow(m, 3.5) * dz ) + 1.0;
      
            float r = length(w);
            float b = 8.0 * acosf(w.y / r);
            float a = 8.0 * atan2f(w.x, w.z);
            w = pos + pow(r, 8.0) * vec3( sinf(b) * sinf(a), cosf(b), sinf(b) * cosf(a));
            m = dot(w , w);
            if(m > 256.0){
                break;
            }
        }
        float d = 0.25 * log(m) * sqrt(m) / dz;

        return d;
    }
}