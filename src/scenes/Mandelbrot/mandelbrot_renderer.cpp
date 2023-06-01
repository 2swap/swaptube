#pragma once

#include "mandelbrot.cpp"
#include "palette.cpp"

class MandelbrotRenderer {
    Mandelbrot m;
    Palette palette;
    int time = 0;
    public:
        enum WhichParameterization{
            Z,
            X,
            C
        };
        Complex z0 = Complex(0, 0);
        Complex x = Complex(0, 0);
        Complex c = Complex(0, 0);

        WhichParameterization which;

        Complex center = Complex(0, 0);
        Complex zoom_multiplier = Complex(0.95, 0.01);
        Complex current_zoom = Complex(.01, 0);
        double breath = 0;

        MandelbrotRenderer() : z0(0.0, 0.0), x(0.0, 0.0), c(0.0, 0.0), which(C), center(0.0, 0.0), current_zoom(0.0, 0.0), zoom_multiplier(0.0, 0.0) {}

        MandelbrotRenderer(Complex _z0, Complex _x, Complex _c, WhichParameterization wp, Complex cnt, Complex cz, Complex zm){
            center = cnt;
            current_zoom = cz;
            zoom_multiplier = zm;
            z0 = _z0;
            x = _x;
            c = _c;
            which = wp;
        }

        void render_mandelbrot(Pixels& p){
            Complex screen_center = Complex(p.w*0.5, p.h*0.5);
            vector<int> depths(p.w*p.h, 12345);
            int queries = 0;
            for(int jump = 16; jump >= 1; jump/=2){
                for(int dx = 0; dx < p.w; dx+=jump)
                    for(int dy = 0; dy < p.h; dy+=jump) {
                        if(depths[dy*p.w+dx] != 12345) continue;
                        Complex point = (Complex(dx, dy)-screen_center) * current_zoom + center;
                        int depth = 0;
                        if (which == Z)
                            depth = m.get_depth_complex(point, x, c);
                        if (which == X)
                            depth = m.get_depth_complex(z0, point, c);
                        if (which == C)
                            depth = m.get_depth_complex(z0, x, point);
                        depths[dx+p.w*dy] = depth;
                        queries++;
                    }
                if(jump != 1){
                    for(int dx = jump*2; dx < p.w-jump*2; dx+=jump)
                        for(int dy = jump*2; dy < p.h-jump*2; dy+=jump){
                            int depth = depths[dx+p.w*dy];
                            bool uniform = true;
                            for(int x = dx-jump*2; x <= dx+jump*2; x+=jump)
                                for(int y = dy-jump*2; y <= dy+jump*2; y+=jump)
                                    if(depths[x+p.w*y]!=depth)
                                        uniform = false;
                            if(uniform)
                                for(int x = dx-jump; x <= dx+jump; x++)
                                    for(int y = dy-jump; y <= dy+jump; y++)
                                        depths[x+p.w*y] = depth;
                        }
                }
            }
            cout << queries << endl;
            //change depths to colors
            for(int dx = 0; dx < p.w; dx++)
                for(int dy = 0; dy < p.h; dy++){
                    int depth = depths[dx+p.w*dy];
                    int col = depth == -1 ? 0 : palette.prompt(depth+time*breath);
                    p.set_pixel(dx, dy, col);
                }
            current_zoom = current_zoom * zoom_multiplier;
            time++;
        }
};
