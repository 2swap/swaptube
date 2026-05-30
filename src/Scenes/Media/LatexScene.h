#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/Interpolation.h"

class LatexScene : public Scene {
public:
    LatexScene(const string& l, double box_scale, const vec2& dimensions = vec2(1, 1));

    void begin_latex_transition(const TransitionType tt, const string& l);

    void jump_latex(string l);

    const StateQuery populate_state_query() const;

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    void draw() override;

private:
    Pixels last_pixels;
    Pixels next_pixels;
    unsigned int last_num_glyphs;
    unsigned int next_num_glyphs;
    Interpolation interp;
    TransitionType transition_type;
    double scale_factor = 0;
    double box_scale;
    bool transitioning = false;
};
