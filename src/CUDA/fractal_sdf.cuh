#pragma once

#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"

namespace sdf{
    
    enum{
        MANDELBULB
    };
    // Computes Signed Distance Function to the Mandelbulb fractal (degree 8)
    // Based off work by Inigo Quilez: https://iquilezles.org/articles/mandelbulb/
    __device__ __forceinline__ float mandelbulb(const Cuda::vec3 pos, int maxIters){ 
        Cuda::vec3 w = pos;
        float dz = 1.0;
        float m = dot(w, w);

        for(int i = 0; i < maxIters; i++){
            #if 1
            
            float invr = 1/sqrtf(m);
            float r4 = m * m;
            float r8 = r4 * r4;
            dz = (8.0 * r8 * invr * dz ) + 1.0;

            float b = 8.0 * acosf(w.y * invr);
            float a = 8.0 * atan2f(w.x, w.z);
            float sinfa, sinfb, cosfa, cosfb;
            sincosf(a, &sinfa, &cosfa);
            sincosf(b, &sinfb, &cosfb);

            w = pos + r8 * Cuda::vec3(sinfb * sinfa, cosfb, sinfb * cosfa);
            #else
            float m2 = m*m;
        float m4 = m2*m2;
		dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;

        float x = w.x; float x2 = x*x; float x4 = x2*x2;
        float y = w.y; float y2 = y*y; float y4 = y2*y2;
        float z = w.z; float z2 = z*z; float z4 = z2*z2;

        float k3 = x2 + z2;
        float k2 = 1/sqrtf( k3*k3*k3*k3*k3*k3*k3 );
        float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;
        float k4 = x2 - y2 + z2;

        w.x = pos.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;
        w.y = pos.y + -16.0*y2*k3*k4*k4 + k1*k1;
        w.z = pos.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2 + 70.0*x4*z4 - 28.0*x2*z2*z4 + z4*z4)*k1*k2;
            #endif

            m = dot(w, w);
            if(m > 256.0) break;
        }
        float d = 0.25 * log(m) * sqrt(m) / dz;

        return d;
    }

}
