#pragma once

#include "../Common/CoordinateScene.h"
#include <vector>
#include <string>

class RealFunctionScene : public CoordinateScene {
public:
    RealFunctionScene(const double width = 1, const double height = 1);

    // Evaluates the function at x by replacing '{a}' with the x-value,
    // computing the current function and its transitional variant, then smoothing between them.
    float call_the_function(const float x, const ResolvedStateEquation& re);

    void draw() override;

    void render_functions();

    const StateQuery populate_state_query() const override;

private:
    std::vector<unsigned int> function_colors = {0xFF0088FF, 0xFFFF0088};
};
