

__device__ __forceinline__ Cuda::vec4 four_d_mult(Cuda::vec4 a, Cuda::vec4 b, float lerp, Cuda::vec4 jj, Cuda::vec4 ijj) { // bool commute) {


    return Cuda::vec4( 
        a.x*b.x - a.y*b.y,
        a.x*b.y + a.y*b.x,
        a.x*b.z + a.z*b.x - lerp*a.w*b.y - a.y*b.w,
        a.x*b.w + a.w*b.x + a.y*b.z + lerp*a.z*b.y
    )
    + (a.z*b.z - lerp*a.w*b.w)*jj
    + (a.w*b.z + lerp*a.z*b.w)*ijj
    ;


    Cuda::vec4 vec_out;
    if (lerp == 1.0){

        // bicomplex
        vec_out = Cuda::vec4( 
            a.x*b.x - a.y*b.y - a.z*b.z + a.w*b.w,
            a.x*b.y + a.y*b.x - a.z*b.w - a.w*b.z,
            a.x*b.z + a.z*b.x - a.w*b.y - a.y*b.w,
            a.x*b.w + a.w*b.x + a.y*b.z + a.z*b.y
        );

    } else if (lerp == 0.0){

        // quaternion
        vec_out = Cuda::vec4( 
            a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
            a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
            a.x*b.z + a.z*b.x + a.w*b.y - a.y*b.w,
            a.x*b.w + a.w*b.x + a.y*b.z - a.z*b.y
        );
    } else {
        return four_d_mult(a,b,0.0,jj,ijj)*(1-lerp) + four_d_mult(a,b,1.0,jj,ijj)*lerp;
    }


    return vec_out;
}




__device__ __forceinline__ Cuda::vec4 four_d_sin(const Cuda::vec4& v, const float commute, const Cuda::vec4& jj, const Cuda::vec4& ijj) {

    if (abs(v.x) > 30){
        return Cuda::vec4(100000000,100000000,100000000,100000000) ;
    }

    Cuda::vec4 sinv = v;
    Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
    Cuda::vec4 v_pow = v;

    for (int t = 3; t < 60; t+=2){
        v_pow = four_d_mult(v_pow, v2,commute,jj,ijj)/((1.0-t)*t);
        sinv += v_pow;  
        if (abs(v_pow.x) + abs(v_pow.y) + abs(v_pow.z) + abs(v_pow.w)> 1000000){
            return Cuda::vec4(100000000,100000000,100000000,100000000) ;
        }
    }
    // if (to_print){
    //     printf("%f %f %f %f\n",sinv.x,sinv.y,sinv.z,sinv.w);
    // }
    return sinv;
}

__device__ __forceinline__ Cuda::vec4 four_d_cos(const Cuda::vec4& v, const float commute, const Cuda::vec4& jj, const Cuda::vec4& ijj) {

    if (abs(v.x) > 30){
    return Cuda::vec4(100000000,100000000,100000000,100000000) ;
    }

    Cuda::vec4 cosv(1,0,0,0);
    Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
    Cuda::vec4 v_pow = v;

    for (int t = 2; t < 60; t+=2){
        v_pow = four_d_mult(v_pow, v2,commute,jj,ijj)/((1.0-t)*t);
        cosv = cosv + v_pow;
        if (abs(v_pow.x) + abs(v_pow.y) + abs(v_pow.z) + abs(v_pow.w)> 1000000){
            return Cuda::vec4(100000000,100000000,100000000,100000000) ;
        }
    }
    return cosv;
}

__device__ __forceinline__ Cuda::vec4 four_d_exp(const Cuda::vec4& v, const float commute, const Cuda::vec4& jj, const Cuda::vec4& ijj) {

    if (abs(v.x) > 30){
        return Cuda::vec4(100000000,100000000,100000000,100000000) ;
    }

    Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
    Cuda::vec4 v3 = four_d_mult(v,v2,commute,jj,ijj);
    Cuda::vec4 v4 = four_d_mult(v2,v2,commute,jj,ijj);
    if (abs(v4.x) + abs(v4.y) + abs(v4.z) + abs(v4.w) > 1000000){
        return Cuda::vec4(100000000,100000000,100000000,100000000) ;
    }
    Cuda::vec4 v5 = four_d_mult(v3,v2,commute,jj,ijj);
    Cuda::vec4 v6 = four_d_mult(v4,v2,commute,jj,ijj);
    Cuda::vec4 v7 = four_d_mult(v5,v2,commute,jj,ijj);
    Cuda::vec4 v8 = four_d_mult(v6,v2,commute,jj,ijj);
    return 1 + v + v2/2 + v3/6 + v4/24 + v5/120 + v6/720 + v7/5040 + v8/40320 ;//+ v9/362880 + v10/3628800;
}


__device__ __forceinline__ float smallishness(float s, float brightness){
    return 1/(1+s*s/brightness);
}


__device__ __forceinline__ float smallness(float s, float brightness){
    // return 1/(1+0.02*s*s*abs(s));
    float s_sq = s*s;
    return 1/(1+s_sq*s_sq/brightness);
    // return 1/(1+s*s*s*s);
}

__device__ __forceinline__ float discrete_smallness(float s, float brightness){
    
    int l = (int) (log(s*s+0.99)/16)*256;
    float sq = pow(2.718,l)-0.35;

    // float s_sq = s*s;
    return 1/(1+sq*sq/brightness);
    // return 1/(1+s*s*s*s);


}


__device__ __forceinline__ Cuda::vec4 four_d_real(float r){
   Cuda::vec4 vec_out(r,0,0,0);
   return vec_out;
}



__device__ __forceinline__ Cuda::vec4 four_d_function(const Cuda::vec4& v, const int equation, const float commute, const Cuda::vec4& jj, const Cuda::vec4& ijj) {

    if (abs(v.x) > 30){
        return Cuda::vec4(100000000,100000000,100000000,100000000) ;
    }

    if (equation == 1){
        return four_d_sin(v, commute, jj, ijj);

    } else if (equation == 10){
        return four_d_cos(v, commute, jj, ijj);

    } else if (equation == 100){
        return four_d_exp(v, commute, jj, ijj); 

    } else if (equation == 4){
        return four_d_mult(v,v,commute,jj,ijj);
        
    } 
    
    Cuda::vec4 v2 = four_d_mult(v,v,commute,jj,ijj);
    Cuda::vec4 v3 = four_d_mult(v,v2,commute,jj,ijj);
    Cuda::vec4 v4 = four_d_mult(v2,v2,commute,jj,ijj);
    
    if (equation == 2){
        return four_d_sin(v4*0.001-v2*0.1+1, commute, jj, ijj);
    } else if (equation == 3){
        return four_d_exp(four_d_sin(v*0.15, commute, jj, ijj)*5, commute, jj, ijj);
    }
    
    Cuda::vec4 v5 = four_d_mult(v3,v2,commute,jj,ijj);
    Cuda::vec4 v6 = four_d_mult(v4,v2,commute,jj,ijj);
    Cuda::vec4 v7 = four_d_mult(v5,v2,commute,jj,ijj);
    Cuda::vec4 v8 = four_d_mult(v6,v2,commute,jj,ijj);
    Cuda::vec4 v9 = four_d_mult(v7,v2,commute,jj,ijj);
    Cuda::vec4 v10 = four_d_mult(v5,v5,commute,jj,ijj);
    Cuda::vec4 v11 = four_d_mult(v5,v6,commute,jj,ijj);
    Cuda::vec4 v12 = four_d_mult(v7,v5,commute,jj,ijj);

    if (equation == 5){
        return v3+1;

    } else if (equation == 6){
        return v4+1;

    } else if (equation == 7){

        // return four_d_function(four_d_function(v, 1, commute, jj, ijj), 0, commute, jj, ijj);
        // Equation 0 is the identity, behind the same escape check.
        if (abs(v2.x) > 30){
            return Cuda::vec4(100000000,100000000,100000000,100000000) ;
        }
        return v2;


    } else if (equation == 101){
        // return four_d_cos(v2*0.06, commute, jj, ijj) + four_d_sin(v*0.4, commute, jj, ijj)*2;

        return four_d_exp(four_d_sin(v*0.3, commute, jj, ijj)*5, commute, jj, ijj)-2;

    } else if (equation == 102){
        // return four_d_cos(v2*0.06, commute, jj, ijj) + four_d_sin(v*0.4, commute, jj, ijj)*2;

        // Cuda::vec4 root5 = v5/512 - Cuda::vec4(0,3,0,0);
        // Cuda::vec4 root9 = v9/1600000 - v3/32 - Cuda::vec4(0,0,4,0);
        // return four_d_mult(root5,root9,commute,jj,ijj);
        // return v7/2048 - 4;
        // return 1 - v + v3 - v4 + v5 - v7 + v8;
        // return v + v3/3 + v5*0.13333 + v7*0.053968 + v9*0.021869 + v11*0.008863;

        // float d = 1.8;

        Cuda::vec4 va = v*0.2;

        Cuda::vec4 p = va - Cuda::vec4(0,1,0,0)*0.06;
        p = four_d_mult(p,va - Cuda::vec4(0,-1,0.3,0),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(0.5,0,0,0.5),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(0,-2,0,0),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(0,0,1,3),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(0.5,0.5,-0.5,0.4),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(3,0,0,0),commute,jj,ijj);
        p = four_d_mult(p,va - Cuda::vec4(0.2,0,-1,0),commute,jj,ijj);

        return p;
    }
    
    return v;

}
