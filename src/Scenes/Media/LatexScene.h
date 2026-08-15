#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/Interpolation.h"

class LatexScene : public Scene {
public:
    LatexScene(const string& l, const vec2& dimensions = vec2(1, 1));

    void begin_latex_transition(const TransitionType tt, const string& l);

    void jump_latex(string l);

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    void draw() override;

private:
    shared_ptr<DevicePointer> last_pixels;
    shared_ptr<DevicePointer> next_pixels;
    unsigned int last_num_glyphs;
    unsigned int next_num_glyphs;
    Interpolation interp;
    TransitionType transition_type;
    bool transitioning = false;
};
