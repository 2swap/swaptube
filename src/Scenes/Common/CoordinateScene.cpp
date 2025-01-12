#pragma once

#include "../Scene.cpp"
#include <vector>

string truncate_tick(double value) {
    // Convert double to string with a stream
    ostringstream oss;
    oss << value;
    string str = oss.str();

    // Find the position of the decimal point
    size_t decimalPos = str.find('.');

    // If there's no decimal point, just return the string
    if (decimalPos == string::npos) {
        return str;
    }

    // Remove trailing zeros
    size_t endPos = str.find_last_not_of('0');

    // If the last non-zero character is the decimal point, remove it too
    if (endPos == decimalPos) {
        endPos--;
    }

    // Create a substring up to the correct position
    return str.substr(0, endPos + 1);
}

class CoordinateScene : public Scene {
public:
    CoordinateScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    void draw() override {
        render_axes();
    }

    void render_axes() {
        int w = get_width();
        int h = get_height();
        double z = state["zoom"];
        double cx = state["center_x"];
        double cy = state["center_y"];
        double sx = cx - .5/z;
        double sy = cy - .5/z;
        double ex = cx + .5/z;
        double ey = cy + .5/z;
        double log10z = log10(z);
        int order_mag = floor(-log10z);
        bool not_fiveish = log10z - floor(log10z) < .5;
        for(int d_om = 0; d_om < 2; d_om++){
            double increment = pow(10, order_mag) * (not_fiveish ? 1 : 0.5);
            double tick_length = (2-d_om) *5; 
            for(double x = floor(sx/increment)*increment; x < ex; x += increment) {
                double frac = (x - sx) / (ex - sx);
                int x_pix = frac * w;
                pix.bresenham(x_pix, 0, x_pix, tick_length, OPAQUE_WHITE, 1, 1);
                ScalingParams sp(40, 20);
                if(d_om == 0){
                    Pixels latex = latex_to_pix(truncate_tick(x), sp);
                    pix.overlay(latex, x_pix, tick_length);
                }
            }
            for(double y = floor(sy/increment)*increment; y < ey; y += increment) {
                double frac = (y - sy) / (ey - sy);
                int y_pix = frac * h;
                pix.bresenham(0, y_pix, tick_length, y_pix, OPAQUE_WHITE, 1, 1);
                ScalingParams sp(40, 20);
                if(d_om == 0){
                    Pixels latex = latex_to_pix(truncate_tick(y), sp);
                    pix.overlay(latex, tick_length, y_pix);
                }
            }
            if(!not_fiveish) order_mag--;
            not_fiveish = !not_fiveish;
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"center_x", "center_y", "zoom"};
    }

    void mark_data_unchanged() override {}
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void on_end_transition(){}
};

