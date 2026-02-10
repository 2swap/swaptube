#pragma once

#include "../Common/CoordinateScene.cpp"
#include <vector>
#include <stdexcept>
#include <string>

class LineChartScene : public CoordinateScene {
public:
    LineChartScene(const double width = 1, const double height = 1) : CoordinateScene(width, height) {
        manager.set({
            {"ticks_opacity", "1"},
            {"left_x", "-.1"},
            {"right_x", "1"},
            {"bottom_y", "-.1"},
            {"top_y", "1"},
            {"window_height", "1 <bottom_y> <top_y> - /"},
        });
    }

    float number_to_add = 0;
    bool ready_to_add = false;

    vector<float> data_points;

    float get_max_data_point() const {
        if(data_points.empty())
            return 0;
        float max_val = data_points[0];
        for(const float& val : data_points) {
            if(val > max_val)
                max_val = val;
        }
        return max_val;
    }

    float get_min_data_point() const {
        if(data_points.empty())
            return 0;
        float min_val = data_points[0];
        for(const float& val : data_points) {
            if(val < min_val)
                min_val = val;
        }
        return min_val;
    }

    void on_end_transition_extra_behavior(const TransitionType tt) override {
        if(ready_to_add) {
            ready_to_add = false;
            data_points.push_back(number_to_add);
        }
    }

    void add_data_point(TransitionType tt, float y) {
        number_to_add = y;
        ready_to_add = true;
        float min_val = min(get_min_data_point(), y) - 1;
        float max_val = max(get_max_data_point(), y) + 1;
        float range = max_val - min_val;
        min_val = min_val - range * .1f;
        max_val = max_val + range * .1f;
        manager.transition(tt, {
            {"left_x", "-1"},
            {"right_x", to_string(data_points.size() + 1)},
            {"bottom_y", to_string(max_val)},
            {"top_y", to_string(min_val)},
        });
    }

    void draw() override {
        render_chart();
        CoordinateScene::draw();
    }

    void render_chart() {
        double left_bound  = state["left_x"];
        double right_bound = state["right_x"];

        float thickness = get_geom_mean_size() / 200.f;

        glm::vec2 last_pixel = {-1, -1};
        bool first_point = true;
        for (int x = 0; x < data_points.size(); x++) {
            float val = data_points[x];
            glm::vec2 pixel = point_to_pixel({x, val});

            string val_string = float_to_pretty_string(val);
            ScalingParams sp(get_width()/10., get_height()/10.);
            Pixels latex = latex_to_pix(val_string, sp);
            pix.overlay(latex, pixel.x - latex.w*.5, pixel.y - latex.h*1.1, 1.0f);

            if (first_point) {
                first_point = false;
                last_pixel = pixel;
                continue;
            }

            pix.bresenham(pixel.x, pixel.y, last_pixel.x, last_pixel.y, 0xff00ffff, 1.0f, thickness);

            last_pixel = pixel;
        }
    }
};

