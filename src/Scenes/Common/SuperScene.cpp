#pragma once

#include "../Scene.cpp"

class SuperScene : public Scene {
public:
    SuperScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    // Scenes which contain other scenes use this to populate the StateQuery
    virtual bool subscene_needs_redraw() const = 0;
    bool needs_redraw() const override {
        bool state_change = check_if_state_changed();
        bool data_change = check_if_data_changed();
        bool subscene_change = subscene_needs_redraw();
        return !has_ever_rendered || state_change || data_change || subscene_change;
    }
};
