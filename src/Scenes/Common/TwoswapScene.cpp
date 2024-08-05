#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class TwoswapScene : public Scene {
public:
    TwoswapScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {
        ScalingParams sp(pix.w*.7, pix.h);
        Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), sp);
        pix.fill_ellipse(pix.w/4, pix.h/2, pix.w/14, pix.w/14, OPAQUE_WHITE);
        pix.overwrite(twoswap_pix, pix.w/4+pix.w/14+pix.w/48, (pix.h-twoswap_pix.h)/2+pix.w/48);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    bool update_data_objects_check_if_changed() override { return false; }
    void draw() override{ }

private:
};
