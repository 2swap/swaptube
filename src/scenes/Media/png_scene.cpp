#pragma once

#include "../../io/visual_media.cpp"
#include "../scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string picture_name) : Scene(1,1) {
        cout << "rendering png: " << picture_name << endl;
        pix = png_to_pix(picture_name);
        resize(pix.w, pix.h);
    }

    void query(Pixels*& p) override {
        p = &pix;
    }

private:
    // nothing yet
};
