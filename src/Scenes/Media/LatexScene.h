#pragma once

#include "../Scene.h"
#include "../../Host_Device_Shared/Interpolation.h"
#include "../../IO/ScalingParams.h"

class LatexScene : public Scene {
public:
    LatexScene(const string& l, const vec2& dimensions = vec2(1, 1));
    ~LatexScene();

    // Owns GPU memory through interp; copying would double-free it.
    LatexScene(const LatexScene&) = delete;
    LatexScene& operator=(const LatexScene&) = delete;

    void begin_latex_transition(const TransitionType tt, const string& l);

    void jump_latex(string l);

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    void draw() override;

private:
    void release_interpolation();

    shared_ptr<DevicePointer> last_pixels;
    shared_ptr<DevicePointer> next_pixels;
    unsigned int last_num_glyphs;
    unsigned int next_num_glyphs;
    Interpolation interp{}; // Zero-initialized so release_interpolation is safe before the first transition
    TransitionType transition_type;
    bool transitioning = false;
    string current_latex; // Equation currently shown (the start of the next transition)
    ScalingParams current_scaling = ScalingParams(vec2(0, 0));
};
