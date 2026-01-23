#pragma once

#include <algorithm>
#include "../../DataObjects/BouncingBalls.cpp"

class BouncingBallScene : public CoordinateScene {
private:
    BouncingBalls balls;

public:
    BouncingBallScene(
        const int n,
        const double simulation_width = 10,
        const double simulation_height = 10,
        const double width = 1,
        const double height = 1
    )
        :CoordinateScene(width, height), balls(n, simulation_width, simulation_height, .05f) {}

    void mark_data_unchanged() override { balls.mark_unchanged(); }
    void change_data() override { balls.iterate(); }
    bool check_if_data_changed() const override { return balls.has_been_updated_since_last_scene_query(); }

    void draw() override{
        for(const Ball& ball : balls.balls){
            glm::vec2 pixel = point_to_pixel({ball.x, ball.y});
            glm::vec2 rim = point_to_pixel({ball.x + balls.ball_radius, ball.y});
            pix.fill_circle(pixel.x, pixel.y, rim.x - pixel.x, 0xFFFFFFFF);
        }
    }
};
