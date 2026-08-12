#pragma once

#include <unordered_map>
#include <chrono>
#include "../Host_Device_Shared/vec.h"
#include "../Core/State/StateManager.h"
#include "../Core/Pixels.h"
#include "../Core/Macroblock.h"
#include "../DataObjects/DevicePointer.h"
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;

extern int remaining_microblocks_in_macroblock;
extern int remaining_frames_in_macroblock;
extern int total_microblocks_in_macroblock;
extern int total_frames_in_macroblock;

void stage_macroblock(const Macroblock& macroblock, int expected_microblocks_in_macroblock);

class Scene {
public:
    Scene(const vec2& dimensions = vec2(1, 1));

    virtual void draw() = 0;

    virtual void on_end_transition_extra_behavior(const TransitionType tt){};
    void on_end_transition(const TransitionType tt);

    void update();

    uint32_t* query();

    double get_geom_mean_size();
    int get_width();
    int get_height();
    ivec2 get_width_height();
    int get_pixels_size();

    void render_microblock();

    void update_state();

    void export_frame(const string& filename, int scaledown = 1);

    StateManager manager;

    void set_global_identifier(const string& id);

    virtual void change_data();

protected:
    DevicePointer gpu_pix;
    StateReturn state;

private:
    string global_identifier = ""; // This is prefixed before the published global state elements
                                   // to uniquely identify this scene if necessary.
                                   // Empty by default, meaning no state is published.

    bool has_updated_since_last_query = false;

    virtual unordered_map<string, double> stage_publish_to_global() const { return unordered_map<string, double>(); }
    void publish_global();

    void render_one_frame(int microblock_frame_number, int scene_duration_frames);
};
