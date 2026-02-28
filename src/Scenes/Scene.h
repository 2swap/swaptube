#pragma once

#include <unordered_map>
#include <chrono>
#include "../Host_Device_Shared/vec.h"
#include "../Core/State/StateManager.h"
#include "../Core/Pixels.h"
#include "../Core/Macroblock.h"
#include "../IO/VisualMedia.h"
#include "../IO/DebugPlot.h"
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
    Scene(const double width = 1, const double height = 1);

    virtual const StateQuery populate_state_query() const = 0;
    virtual bool check_if_data_changed() const = 0;
    virtual void draw() = 0;
    virtual void change_data() = 0;
    virtual void mark_data_unchanged() = 0;

    virtual void on_end_transition_extra_behavior(const TransitionType tt){};
    void on_end_transition(const TransitionType tt);

    void update();

    virtual bool needs_redraw() const;

    bool check_if_state_changed() const;

    void query(Pixels*& p);

    void render_microblock();

    void update_state();

    int get_width() const;

    int get_height() const;

    void export_frame(const string& filename, int scaledown = 1) const;

    StateManager manager;

    void set_global_identifier(const string& id);

protected:
    Pixels pix;
    StateReturn state;
    bool has_ever_rendered = false;

    vec2 get_width_height() const;

    double get_geom_mean_size() const;

private:
    string global_identifier = ""; // This is prefixed before the published global state elements
                                   // to uniquely identify this scene if necessary.
                                   // Empty by default, meaning no state is published.

    StateReturn last_state;
    bool has_updated_since_last_query = false;

    virtual unordered_map<string, double> stage_publish_to_global() const { return unordered_map<string, double>(); }
    void publish_global();

    void render_one_frame(int microblock_frame_number, int scene_duration_frames);
};
