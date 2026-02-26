#pragma once

#include "../Scene.h"
#include "../../Core/Convolution.h"
#include <vector>
#include <utility>

class ConvolutionScene : public Scene {
public:
    ConvolutionScene(const double width = 1, const double height = 1);

    std::pair<int, int> get_coords_from_pixels(const Pixels& p);

    void begin_transition(const TransitionType tt, const Pixels& p);
    void jump(const Pixels& p);
    void on_end_transition_extra_behavior(const TransitionType tt) override;
    void end_transition();
    void draw() override;

    const StateQuery populate_state_query() const override;
    void mark_data_unchanged() override;
    void change_data() override;
    bool check_if_data_changed() const override;

private:
    virtual Pixels get_p1() = 0;
    // Things used for non-transition states
    TransitionType current_transition_type;
    Pixels p1;
    std::pair<int, int> coords;
    std::vector<StepResult> intersections;
    Pixels p2;
    std::pair<int, int> transition_coords;
    bool jumped = false;
};
