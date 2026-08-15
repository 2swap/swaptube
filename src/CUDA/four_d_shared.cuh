

__device__ __forceinline__ Cuda::vec4 four_d_mult(Cuda::vec4 a, Cuda::vec4 b, float lerp) { // bool commute) {

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


        // vec_out = Cuda::vec4( 
        //     a.x*b.x + a.y*b.y + 0.8*a.z*b.z - 0.8*a.w*b.w,
        //     a.x*b.y + a.y*b.x - 0.8*a.z*b.w + 0.8*a.w*b.z,
        //     a.x*b.z + a.z*b.x - a.w*b.y + a.y*b.w,
        //     a.x*b.w + a.w*b.x + a.y*b.z - a.z*b.y
        // );

        // vec_out = Cuda::vec4( 
        //     a.x*b.x + a.y*b.y,
        //     a.x*b.y + a.y*b.x,
        //     a.x*b.z + a.z*b.x - a.w*b.y + a.y*b.w,
        //     a.x*b.w + a.w*b.x + a.y*b.z - a.z*b.y
        // );


        // vec_out = Cuda::vec4( 
        //     a.x*b.x - a.y*b.w - a.z*b.z - a.w*b.y,
        //     a.x*b.y + a.y*b.x - a.z*b.w - a.w*b.z,
        //     a.x*b.z + a.y*b.y + a.z*b.x - a.w*b.w,
        //     a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x
        // );

        // vec_out = Cuda::vec4( 
        //     a.x*b.x + a.y*b.w + a.z*b.z + a.w*b.y,
        //     a.x*b.y + a.y*b.x + a.z*b.w + a.w*b.z,
        //     a.x*b.z + a.y*b.y + a.z*b.x + a.w*b.w,
        //     a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x
        // );


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

// __device__ __forceinline__ float smallerness(float s, float brightness){
//     return 1/(1+log(abs(s)+1)/brightness);
// }

__device__ __forceinline__ float smallness(float s, float brightness){
    // return 1/(1+0.02*s*s*abs(s));
    float s_sq = s*s;
    return 1/(1+s_sq*s_sq/brightness);
    // return 1/(1+s*s*s*s);
}


__device__ __forceinline__ Cuda::vec4 four_d_real(float r){
   Cuda::vec4 vec_out(r,0,0,0);
   return vec_out;
}