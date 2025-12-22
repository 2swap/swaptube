#pragma once

#include <iostream>
#include <sstream>
#include <iomanip>
#include "../../IO/VisualMedia.cpp"
#include "../Scene.cpp"

class PauseScene : public Scene {
public:
    PauseScene(const double width = 1, const double height = 1)
    : Scene(width, height) {
        manager.set(unordered_map<string, string>{
            {"timer", "{microblock_fraction}"},
        });
    }

    void draw() override {
        const int w = get_width();
        const int h = get_height();

        double timer_value = state["timer"]; // expected to be between 0 and 1

        double center_x = w / 2.0;
        double center_y = h / 2.0;

        int color = OPAQUE_WHITE;

        // Draw expanding borders from center of each side outward towards corners
        // At 0 timer_value nothing drawn, at 1 timer_value the fill_rect reaches the corners

        const double border_thickness = get_geom_mean_size() / 200.;

        // Top border (from center of top edge to the corners)
        const double top_width = w * timer_value;
        pix.fill_rect(center_x - top_width/2.,                  0, top_width, border_thickness, color);
        pix.fill_rect(center_x - top_width/2., h-border_thickness, top_width, border_thickness, color);

        // Left border (from center left edge toward top-left and bottom-left corners)
        const double left_height = h * timer_value;
        pix.fill_rect(                 0, center_y - left_height/2., border_thickness, left_height, color);
        pix.fill_rect(w-border_thickness, center_y - left_height/2., border_thickness, left_height, color);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"timer"};
    }
    void mark_data_unchanged() override { }
    void change_data() override { }
    bool check_if_data_changed() const override { return false; } // No DataObjects
};
