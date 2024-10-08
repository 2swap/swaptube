#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class TwoswapScene : public Scene {
public:
    TwoswapScene(const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height) {
        ScalingParams sp(pix.w*.7, pix.h);
        saved_pix = Pixels(width, height);
        Pixels twoswap_pix = latex_to_pix(latex_text("2swap"), sp);
        saved_pix.fill_ellipse(pix.w/4, pix.h/2, pix.w/14, pix.w/14, OPAQUE_WHITE);
        saved_pix.overwrite(twoswap_pix, pix.w/4+pix.w/14+pix.w/48, (pix.h-twoswap_pix.h)/2+pix.w/48);
        ScalingParams sp2(pix.w*.4, pix.h*.2);
        Pixels swaptube_pix = latex_to_pix(latex_text("Rendered with love, using SwapTube"), sp2);
        saved_pix.overwrite(swaptube_pix, pix.w*.95-swaptube_pix.w, pix.h*.95-swaptube_pix.h);
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }
    void mark_data_unchanged() override { }
    void change_data() override {}
    bool check_if_data_changed() const override { return false; }
    void draw() override{pix = saved_pix;}
    void on_end_transition() override{}

private:
    Pixels saved_pix;
};
