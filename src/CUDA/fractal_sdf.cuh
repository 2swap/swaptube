#pragma once

#include <cuda_runtime.h>
#include "../Host_Device_Shared/vec.h"

namespace sdf{
    
    enum{
        MANDELBULB,
        MANDELBULB_8,
        BURNINGBULB,
    };

    __device__ __forceinline__ float mandelbulb(const Cuda::vec3& pos, const float power, const int maxIters){
        Cuda::vec3 w = pos;
        float dz = 1.0f;
        float m = dot(w, w);

        for(int i = 0; i < maxIters; i++){
            float r = sqrtf(m);
            float rp = powf(r, power);
            float invr = rsqrtf(m);
            dz = (power * rp * invr * dz) + 1.0f;

            float b = power * asinf(w.y * invr);
            float a = power * atan2f(w.z, w.x);
            float sinfa, sinfb, cosfa, cosfb;
            sincosf(a, &sinfa, &cosfa);
            sincosf(b, &sinfb, &cosfb);

            w = fabsf(rp) * Cuda::vec3(fabsf(cosfb * cosfa), fabsf(-sinfb), fabsf(cosfb * sinfa)) + pos;

            m = dot(w, w);
            if(m > 256.0f) break;
        }
        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    __device__ __forceinline__ float burningbulb2(const Cuda::vec3& pos, const int maxIters){
        Cuda::vec3 w = pos;
        float dz = 1.0f;
        float m = dot(w, w);

        for(int i = 0; i < maxIters; i++){
            float r = sqrtf(m);
            float invr = rsqrtf(m);
            dz = (2.0 * r * dz) + 1.0f;

            float rxy2 = w.x * w.x + w.y * w.y;
            float g = (1.0f - w.z * w.z / rxy2);

            w.x = fabsf(w.x);
            w.y = fabsf(w.y);
            w.z = fabsf(w.z);

            w = Cuda::vec3((w.x * w.x - w.y * w.y) * g, 2.0f * w.x * w.y * g, -2.0 * w.z * sqrtf(rxy2)) + pos;

            /*float b = 2.0 * asinf(w.y * invr);
            float a = 2.0 * atan2f(w.z, w.x);
            float sinfa, sinfb, cosfa, cosfb;
            sincosf(a, &sinfa, &cosfa);
            sincosf(b, &sinfb, &cosfb);

            w = m * Cuda::vec3(fabsf(cosfb * cosfa), fabsf(-sinfb), fabsf(cosfb * sinfa)) + pos;
*/
            m = dot(w, w);
            if(m > 256.0f) break;
        }
        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    // Computes Signed Distance Function to the Mandelbulb fractal (degree 8)
    // Based off work by Inigo Quilez: https://iquilezles.org/articles/mandelbulb/
    __device__ __forceinline__ float mandelbulb8(const Cuda::vec3& pos, const int maxIters){ 
        Cuda::vec3 w = pos;
        float dz = 1.0f;
        float m = dot(w, w);

        for(int i = 0; i < maxIters; i++){
            float invr = rsqrtf(m);
            float r4 = m * m;
            float r8 = r4 * r4;
            dz = (8.0f * r8 * invr * dz ) + 1.0f;

            float b = 8.0f * acosf(w.y * invr);
            float a = 8.0f * atan2f(w.x, w.z);
            float sinfa, sinfb, cosfa, cosfb;
            sincosf(a, &sinfa, &cosfa);
            sincosf(b, &sinfb, &cosfb);

            w = r8 * Cuda::vec3(sinfb * sinfa, cosfb, sinfb * cosfa) + pos;

            m = dot(w, w);
            if(m > 256.0f) break;
        }
        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    __device__ __forceinline__ float juliabulb(const Cuda::vec3& z, const Cuda::vec3& c, const float power, const int maxIters){
        Cuda::vec3 w = z;
        float dz = 1.0f;
        float m = dot(w, w);

        for(int i = 0; i < maxIters; i++){
            float r = sqrtf(m);
            float rp = powf(r, power);
            float invr = rsqrtf(m);
            dz = (power * rp * invr * dz);

            float b = power * asinf(w.y * invr);
            float a = power * atan2f(w.z, w.x);
            float sinfa, sinfb, cosfa, cosfb;
            sincosf(a, &sinfa, &cosfa);
            sincosf(b, &sinfb, &cosfb);

            w = fabsf(rp) * Cuda::vec3(fabsf(cosfb * cosfa), fabsf(-sinfb), fabsf(cosfb * sinfa)) + c;

            m = dot(w, w);
            if(m > 256.0f) break;
        }
        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    __device__ __forceinline__ float length2(Cuda::quat q){
        return (q.u * q.u + q.i * q.i + q.j * q.j + q.k * q.k);
    }

    __device__ Cuda::quat qSquare(Cuda::quat q)
    {
        return Cuda::quat(q.u*q.u - q.i*q.i - q.j*q.j - q.k*q.k, 2.0*q.u*q.i, 2.0*q.u*q.j, 2.0*q.u*q.k);
    }

    __device__ __forceinline__ float quatJulia2(const Cuda::quat pos, const Cuda::quat c, const int maxIters){
        Cuda::quat z = pos;
        float dz2 = 1.0f;
        float m2 = length2(z);

        for(int i = 0; i < maxIters; i++){
            dz2 *= (4.0f * length2(z));

            z = z * z + c;

            m2 = length2(z);
            if(m2 > 256.0f) break;
        }

        float d = 0.25f * logf(m2) * sqrtf(m2 / dz2);
        
        return d;
    }

    __device__ __forceinline__ float mandelbrot(const float cr, const float ci, const int maxIters){
        float zr = 0.0f;
        float zi = 0.0f;
        float dz = 1.0f;
        float m = 0.0f;

        for(int i = 0; i < maxIters; i++){
            dz = (2.0f * sqrtf(m)) * dz + 1.0f;

            float zr2 = zr * zr;
            float zi2 = zi * zi;

            zi = 2.0f * zr * zi + ci;
            zr = zr2 - zi2 + cr;

            m = zr * zr + zi * zi;
            if(m > 256.0f) break;
        }

        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    __device__ __forceinline__ float julia(float zr, float zi, const float cr, const float ci, const int maxIters){
        float dz = 1.0f;
        float m = 0.0f;

        for(int i = 0; i < maxIters; i++){
            dz = (2.0f * sqrtf(m)) * dz;

            float zr2 = zr * zr;
            float zi2 = zi * zi;

            zi = 2.0f * zr * zi + ci;
            zr = zr2 - zi2 + cr;

            m = zr * zr + zi * zi;
            if(m > 256.0f) break;
        }

        float d = 0.25f * logf(m) * sqrtf(m) / dz;

        return d;
    }

    __device__ __forceinline__ float mandeljulia(float zr, float zi, const float cr, const float ci, const int maxIters){
        float dzc = 1.0f;
        float dz = 1.0f;
        float dc = 1.0f;
        float m = 0.0f;

        for(int i = 0; i < maxIters; i++){
            dz = (2.0f * sqrtf(m)) * dz + 1.0f;
            dc = (2.0f * sqrtf(m)) * dz;

            dzc = sqrtf(dz * dz + dc * dc);

            float zr2 = zr * zr;
            float zi2 = zi * zi;

            zi = 2.0f * zr * zi + ci;
            zr = zr2 - zi2 + cr;

            m = zr * zr + zi * zi;
            if(m > 256.0f) break;
        }

        float d = 0.5f * logf(m) * sqrtf(m) / dzc;

        return d;
    }

}
