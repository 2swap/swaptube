#pragma once

#include "scene.cpp"

class TwoswapScene : public Scene {
public:
    TwoswapScene(const int width, const int height) : Scene(width, height) {init();}
    TwoswapScene() : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {init();}

    void init(){
        Pixels twoswap_pix = eqn_to_pix(latex_text("2swap"), pix.w/160);

        pix.fill_ellipse(pix.w/3, pix.h/2, pix.w/12, pix.w/12, WHITE);
        pix.copy(twoswap_pix, pix.w/3+pix.w/12+pix.w/32, (pix.h-twoswap_pix.h)/2+pix.w/32, 1);
    }

    void query(bool& done_scene, Pixels*& p) override {
        done_scene = scene_duration_frames <= time;

        //Pixels* ret = new Pixels(pix.w, pix.h);
        //cout << "copying pix" << endl;
        //ret->copy(pix, 0, 0, fifo_curve(time / static_cast<double>(scene_duration_frames)));
        //cout << "copied pix" << endl;
        time++;

        p = &pix;
    }
};