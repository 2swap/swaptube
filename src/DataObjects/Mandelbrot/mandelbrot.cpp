#pragma once

//#define BURNING_SHIP 1

#include "complex.cpp"
class Mandelbrot {
    int escape_radius = 2;
    int escape_radius_squared = escape_radius*escape_radius;
    public:
        //How many iterations does it take until z=z^x+c diverges?
        int get_depth(Complex c, int& depth){//, int& out_coords){
            //unpack the complex for speeeeed
            double r0 = c.real;
            double i0 = c.img;
            double r = 0;
            double i = 0;
            double r2 = 0;
            double i2 = 0;
            double w=0;
            for(int d = 0; d < depth; d++){
                r = r2-i2+r0;
                i = w - r2 - i2 + i0;
                r2 = r*r;
                i2 = i*i;
                w=(i+r)*(i+r);
                if(r2+i2 > escape_radius_squared) return d;
            }
            return -1;
        }
        int get_depth_complex(Complex z, const Complex& x, const Complex& c, int& depth){//, int& out_coords){
            if(c == Complex(2, 0)) return get_depth_exp2(z, x, c, depth);
            int d = 0;
            double radius2 = 10000;//pow(2, 1/(x.real-1.0));
            for(d = 0; d < depth; d++){
                if(z.magnitude2() > radius2) return d;
                z = z.pow(x) + c;
                d++;
            }
            if(z.magnitude2() > radius2){
                depth++;
            }
            return d;
        }
        int get_depth_exp2(Complex z, const Complex& x, const Complex& c, int& depth){//, int& out_coords){
            int d = 0;
            for(d = 0; d < depth; d++){
                if(z.magnitude2() > escape_radius_squared) return d;
                z = z*z + c;
                d++;
            }
            if(z.magnitude2() > escape_radius_squared){
                depth++;
            }
            return d;
        }
};
