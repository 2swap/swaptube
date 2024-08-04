#pragma once

#include "LambdaExpression.cpp"

class LambdaScene : public Scene {
public:
    LambdaScene(const string& lambda_str, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
    : Scene(width, height), le(parse_lambda_from_string(lambda_str)), lambda_pixels(le->draw_lambda_diagram()), lambda_string(lambda_str) { }

    void draw() override {
        pix.fill(0xff0080ff);
        pix.overwrite(lambda_pixels, 0, 0);
    }

    bool scene_requests_rerender() const override { return false; }

private:
    // Things used for non-transition states
    LambdaExpression* le;
    Pixels lambda_pixels;
    string lambda_string;
};
