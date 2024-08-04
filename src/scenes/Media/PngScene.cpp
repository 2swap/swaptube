#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string picture_name) : Scene(1,1) {
        cout << "rendering png: " << picture_name << endl;
        pix = png_to_pix(picture_name);
        w = pix.w;
        h = pix.h;
    }

    bool scene_requests_rerender() const override { return false; }
    void draw() override{ }

private:
    // nothing yet
};
