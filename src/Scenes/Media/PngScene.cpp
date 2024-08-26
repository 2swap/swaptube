#pragma once

#include "../../io/VisualMedia.cpp"
#include "../Scene.cpp"

class PngScene : public Scene {
public:
    PngScene(string pn, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT) : Scene(width, height), picture_name(pn) { draw(); }

    bool check_if_data_changed() const override { return false; }
    void mark_data_unchanged() override { }
    void change_data() override { }
    void draw() override{
        cout << "rendering png: " << picture_name << endl;
        pix.overwrite(png_to_pix(picture_name), 0, 0);
    }
    void on_end_transition() override{ }
    const StateQuery populate_state_query() const override {
        return StateQuery{};
    }

private:
    string picture_name;
};
