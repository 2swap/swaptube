#pragma once

#include "../../io/visual_media.cpp"
#include "../Scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string picture_name) : Scene(1,1) {
        cout << "rendering png: " << picture_name << endl;
        pix = png_to_pix(picture_name);
        resize(pix.w, pix.h);
    }

    void draw() override{
    }

private:
    // nothing yet
};
