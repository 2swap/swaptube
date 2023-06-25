#pragma once

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
            int d = 0;
            for(d = 0; d < depth; d++){
                if(z.magnitude2() > escape_radius_squared) return d;
                z = z*z + c;
#ifdef BURNING_SHIP
                z.real = std::abs(z.real);
                z.img = std::abs(z.img);
#endif
                d++;
            }
            if(z.magnitude2() > escape_radius_squared){
                depth++;
            }
            return d;
        }
};
