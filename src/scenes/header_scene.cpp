#pragma once

#include "scene.cpp"

class HeaderScene : public Scene {
public:
    HeaderScene(const int width, const int height, string h, string s) : Scene(width, height), header(h), subheader(s) {
        init();
    }
    HeaderScene(string h, string s) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT), header(h), subheader(s) {
        init();
    }

    void init(){
        Pixels header_pix = eqn_to_pix(latex_text(header), pix.w / 640 + 1);
        Pixels subheader_pix = eqn_to_pix(latex_text(subheader), pix.w / 640);

        pix.fill(BLACK);
        pix.copy(header_pix, (pix.w - header_pix.w)/2, pix.h/2-100, 1);
        pix.copy(subheader_pix, (pix.w - subheader_pix.w)/2, pix.h/2+50, 1);
    }

    Pixels* query(bool& done_scene) {
        done_scene = time >= scene_duration_frames;

        time++;
        return &pix;
    }
private:
    string header;
    string subheader;
};
