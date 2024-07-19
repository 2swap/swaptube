#pragma once

#include "../../misc/visual_media.cpp"
#include "../scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string picture_name) : Scene(VIDEO_WIDTH, VIDEO_HEIGHT) {init_png_scene(picture_name);}

    void init_png_scene(string picture_name){
        cout << "rendering png: " << picture_name << endl;
        pix = png_to_pix(picture_name);
        w = pix.w;
        h = pix.h;
    }

    void query(bool& done_scene, Pixels*& p) override {
        p = &pix;
    }

private:
    // nothing yet
};
