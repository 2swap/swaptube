#pragma once

#include "complex.cpp"
class Mandelbrot {
    int escape_radius = 2;
    int escape_radius_squared = escape_radius*escape_radius;
    public:
        //How many iterations does it take until z=z^x+c diverges?
        int get_depth(Complex c){//, int& out_coords){
            //unpack the complex for speeeeed
            double r0 = c.real;
            double i0 = c.img;
            double r = 0;
            double i = 0;
            double r2 = 0;
            double i2 = 0;
            double w=0;
            for(int d = 0; d < 600; d++){
                r = r2-i2+r0;
                i = w - r2 - i2 + i0;
                r2 = r*r;
                i2 = i*i;
                w=(i+r)*(i+r);
                if(r2+i2 > escape_radius_squared) return d;
            }
            return -1;
        }
        int get_depth_complex(Complex z, Complex x, Complex c){//, int& out_coords){
            int d = 0;
            for(int d = 0; d < 600; d++){
                if(z.magnitude2() > escape_radius*escape_radius) return d;
                z = z*z + c;
#ifdef BURNING_SHIP
                z.real = std::abs(z.real);
                z.img = std::abs(z.img);
#endif
                d++;
            }
            return -1;
        }
};
