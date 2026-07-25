#pragma once

#include "../Common/CoordinateScene.h"
#include "../../DataObjects/Rope.h"


class RopeScene : public CoordinateScene {
    public:
        Rope* rope;
        


        RopeScene(const vec2& dimensions = vec2(1, 1));

        void draw() override;

        const StateQuery populate_state_query() const override;


    private:
        void set_pins(vec2 pos, uint32_t color, float size);
};
