#include <cuda_runtime.h>
#include <vector>
#include "../Host_Device_Shared/vec.h"
#include "../Host_Device_Shared/helpers.h"
#include "common_graphics.cuh"



__device__ float smallerness(float s, float brightness){
    return 1/(1+log(abs(s)+1)/brightness);
}

__device__ float smallness(float s){
    // return 1/(1+0.02*s*s*abs(s));
    // return 1/(1+0.0001*s*s*s*s/brightness);
    return 1/(1+s*s*s*s);
}


__device__ Cuda::vec3 four_d_accum(Cuda::vec4 a, float brightness) {

    // float real_smallness = 4;
    // float real_smallness = 0.01;
    // float real_smallness = 0.006;
    // float real_smallness = 0.01;

    // float real_smallness = (0.8+0.2*smallness(a.x,brightness))*5;
    // float real_smallness = 0.4*smallness(a.x,brightness);
    // real_smallness = real_smallness*real_smallness*real_smallness;
    // float b = smallness(a.y*a.w*a.z*100,brightness);
    Cuda::vec3 accum(
        //  b, b, b
        smallness(a.y/brightness),
        smallness(a.z/brightness),
        smallness(a.w/brightness)
    );
    return accum;

    // float real_smallness = 0.00002;//0.8+0.2*smallness(a.x,brightness);
    // Cuda::vec3 accum(
    //     //  b, b, b
    //     real_smallness*a.y*brightness,
    //     real_smallness*a.z*brightness,
    //     real_smallness*a.w*brightness
    // );
    // return accum;
}

__device__ uint32_t accum_to_color(Cuda::vec3 a) {

    // return 255 << 24 |
    // (uint32_t) min(a.x,255.0) << 16 |
    // (uint32_t) min(a.y,255.0) << 8 |
    // (uint32_t) min(a.z,255.0);

    float b = 0.007;
    float ax = min(1.0,b*a.x);
    float ay = min(1.0,b*a.y);
    float az = min(1.0,b*a.z);

    return Cuda::OKLABtoRGB(
        255,
        min(1.0,(ax+ay+az)*0.5),
        (ax-ay)*0.866,
        (ax+ay)*0.5-az
    );
}


__device__ Cuda::vec4 four_d_mult(Cuda::vec4 a, Cuda::vec4 b, float lerp) { // bool commute) {

    Cuda::vec4 vec_out;
    // quaternion
    if (lerp == 1.0){

        // bicomplex
        vec_out = Cuda::vec4( 
            a.x*b.x - a.y*b.y - a.z*b.z + a.w*b.w,
            a.x*b.y + a.y*b.x - a.z*b.w - a.w*b.z,
            a.x*b.z + a.z*b.x - a.w*b.y - a.y*b.w,
            a.x*b.w + a.w*b.x + a.y*b.z + a.z*b.y
        );
    } else if (lerp == 0.0){

        vec_out = Cuda::vec4( 
            a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
            a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
            a.x*b.z + a.z*b.x + a.w*b.y - a.y*b.w,
            a.x*b.w + a.w*b.x + a.y*b.z - a.z*b.y
        );
    } else {
        return four_d_mult(a,b,0.0)*(1-lerp) + four_d_mult(a,b,1.0)*lerp;
    }


    return vec_out;
}

__device__ Cuda::vec4 four_d_real(float r){
   Cuda::vec4 vec_out(r,0,0,0);
   return vec_out;
}


__device__ Cuda::vec4 four_d_function(Cuda::vec4 v, const int equation, float commute, bool to_print) {


    if (equation == 1){

        Cuda::vec4 sinv = v;
        Cuda::vec4 v2 = four_d_mult(v,v,commute);
        Cuda::vec4 v_pow = v;

        for (int t = 3; t < 30; t+=2){
            v_pow = four_d_mult(v_pow, v2,commute)/(-1.0*(t-1.0)*t);
            sinv += v_pow;
        }
        // if (to_print){
        //     printf("%f %f %f %f\n",sinv.x,sinv.y,sinv.z,sinv.w);
        // }
        return sinv;

    } else if (equation == 2){

        Cuda::vec4 cosv(1,0,0,0);
        Cuda::vec4 v2 = four_d_mult(v,v,commute);
        Cuda::vec4 v_pow = v;

        for (int t = 2; t < 41; t+=2){
            v_pow = four_d_mult(v_pow, v2,commute)/(-1.0*(t-1.0)*t);
            cosv = cosv + v_pow;
        }
        return cosv;

    }

    Cuda::vec4 v2 = four_d_mult(v,v,commute);
    Cuda::vec4 v3 = four_d_mult(v,v2,commute);
    Cuda::vec4 v4 = four_d_mult(v2,v2,commute);

    Cuda::vec4 v5 = four_d_mult(v3,v2,commute);
    Cuda::vec4 v6 = four_d_mult(v4,v2,commute);
    Cuda::vec4 v7 = four_d_mult(v5,v2,commute);
    Cuda::vec4 v8 = four_d_mult(v6,v2,commute);
    Cuda::vec4 v9 = four_d_mult(v7,v2,commute);
    Cuda::vec4 v10 = four_d_mult(v5,v5,commute);



    if (equation == 0){
        return 1 + v + v2/2 + v3/6 + v4/24 + v5/120 + v6/720 + v7/5040 + v8/40320;

    } else if (equation == 3){
        return v - v2 - v5 + v10;
        // return -2 + v*6 - v2*2 - v3*3 + v6;
        // return 1 + v + v2 + v3 + v4;
        // return 1 - v + v3 - v4 + v5 - v7 + v8;
        // return 1 - v + v2 - v3 + v4;

    }
    
    return v;

    // Cuda::vec4 v11 = four_d_mult(v9,v2);
    // Cuda::vec4 v15 = four_d_mult(v9,v6);


    // Cuda::vec4 m1 = v4+four_d_real(3);
    // m1 = four_d_mult( m1, v2-four_d_real(4),commute);
    // m1 = four_d_mult( m1, v3+four_d_real(3),commute);
    // return m1;

    // return v4;


    // Cuda::vec4 coeff(0.2,1.0,0.2,-1.4);
    // Cuda::vec4 cv4 = four_d_mult(coeff,cv4);
    // return v7+4.0*v6-10*v5+20*v-16.0;
    // return v15-7.0*v5-5.0*v7+four_d_real(35.0);
    // return v6-v4+10.0*v3+1.0;
    // return v18*0.01 - v7 + v6*2.7 +  v5*8.0 - v3*60.0 -50.0;
    // return sinv*0.8 + cosv*1.6;
    // return sinv;
}

__global__ void four_d_raymarch_kernel(
    const Cuda::ivec2 wh,

    Cuda::quat camera_orientation,
    Cuda::vec3 camera_position,
    float fov, float max_dist,

    const Cuda::vec4 x_unit,
    const Cuda::vec4 y_unit,
    const Cuda::vec4 z_unit,
    const float brightness,
    const float slider,
    const int equation,

    uint32_t* colors
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixel_x >= wh.x || pixel_y >= wh.y) return;

    Cuda::ivec2 pixel(pixel_x, pixel_y);
    // Cuda::vec2 point_vec = pixel_to_point_in_screen(pixel, lx_ty, rx_by, wh);

    Cuda::vec3 out(0.0,0.0,0.0);
    float dist_traveled = 0.0f;
    float dt = 0.01f;
    
    Cuda::vec3 dir_world = normalize(Cuda::get_raymarch_vector(pixel, wh, fov, camera_orientation))*dt;

    Cuda::vec3 current_position = camera_position + dir_world;

    const float commute = min(max(0.5+(float(pixel_x)/float(wh.x)-slider)*20.0,0.0),1.0);

    while (dist_traveled < max_dist) {
        dist_traveled += dt;
        current_position += dir_world;
        

        // Cuda::vec4 four_d_input(0,current_position.x, current_position.y, current_position.z);
        // Cuda::vec4 four_d_output = four_d_function(four_d_input);
        Cuda::vec4 four_d_output = four_d_function(x_unit*current_position.x+y_unit*current_position.y + z_unit*current_position.z, equation, commute, pixel_x==pixel_y && pixel_y==0);

        out += four_d_accum(four_d_output,brightness); 
    }

    colors[pixel_y * wh.x + pixel_x] = accum_to_color(out); 

}

// Host function to launch the kernel
extern "C" void four_d_render(
    const Cuda::ivec2& wh,

    const Cuda::quat& camera_orientation, 
    const Cuda::vec3& camera_position,
    float fov_rad, 
    float max_dist,

    Cuda::vec4 x_unit,
    Cuda::vec4 y_unit,
    Cuda::vec4 z_unit,

    const float brightness,
    const float slider,
    const int equation,

    uint32_t* d_colors
) {
    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);  // 2D block of 16x16 threads
    dim3 numBlocks((wh.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (wh.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    four_d_raymarch_kernel<<<numBlocks, threadsPerBlock>>>(
        wh,
        camera_orientation, camera_position,
        fov_rad, max_dist,
        x_unit, y_unit,z_unit,

        brightness, 
        slider,
        equation,

        d_colors
    );
    cudaDeviceSynchronize();
}
