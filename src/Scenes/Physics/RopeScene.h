#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/Rope.h"


class RopeScene : public CoordinateScene {
public:
    Rope rope;
    
    RopeScene(const string file_name, const vec2& dimensions = vec2(1, 1));

    void draw() override;

    void add_pin(vec2 pos);
    void remove_pin(int pin_index);
    void change_data();

private:
    void set_pins(vec2 pos, uint32_t color, float size);
};
