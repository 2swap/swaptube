#pragma once

#include "../../IO/Writer.h"
#include "../Scene.h"

// The idea here is that the user does whatever they want with the Pixels object, manually
class ExposedPixelsScene : public Scene {
public:
    Pixels exposed_pixels;
    ExposedPixelsScene(const vec2& dimensions);

    const StateQuery populate_state_query() const override;
    void draw() override;
};
