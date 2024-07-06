#pragma once

#include "../misc/visual_media.cpp"
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
        ScalingParams header_sp(pix.w, pix.h/4);
        Pixels header_pix = eqn_to_pix(latex_text(header), header_sp);
        ScalingParams subheader_sp(pix.w, pix.h/6);
        Pixels subheader_pix = eqn_to_pix(latex_text(subheader), subheader_sp);

        pix.overwrite(header_pix, (pix.w - header_pix.w)/2, pix.h/2-100);
        pix.copy(subheader_pix, (pix.w - subheader_pix.w)/2, pix.h/2+50);
    }

    void query(bool& done_scene, Pixels*& p) override {
        p = &pix;
    }
private:
    string header;
    string subheader;
};
