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

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }
    void draw() override{ }

private:
    // nothing yet
};
