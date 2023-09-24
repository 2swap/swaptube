#pragma once

#include "scene.cpp"

class TwoswapScene : public Scene {
public:
    Scene* createScene(const int width, const int height) override {
        return new TwoswapScene(width, height);
    }

    TwoswapScene(const int width, const int height) : Scene(width, height) {
        Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), pix.w/160);

        pix.fill(BLACK);
        pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, WHITE);
        pix.copy(twoswap_pix, pix.w/3+pix.w/12+pix.w/32, (pix.h-twoswap_pix.h)/2+pix.w/32, 1);
    }

    const Pixels& query(bool& done_scene) override {
        done_scene = scene_duration_frames <= time;

        Pixels ret(pix.w, pix.h);
        ret.fill(BLACK);
        ret.copy(pix, 0, 0, fifo_curve(time / static_cast<double>(scene_duration_frames)));
        time++;

        return ret;
    }
};