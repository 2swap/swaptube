#include "BouncingBallScene.h"

BouncingBallScene::BouncingBallScene(
    const int n,
    const double simulation_width,
    const double simulation_height,
    const vec2& dimensions
)
    : CoordinateScene(dimensions), balls(n, simulation_width, simulation_height, .05f) {}

void BouncingBallScene::draw(){
    for(const Ball& ball : balls.balls){
        vec2 pixel = point_to_pixel({ball.x, ball.y});
        vec2 rim = point_to_pixel({ball.x + balls.ball_radius, ball.y});
        pix.fill_circle(pixel.x, pixel.y, rim.x - pixel.x, 0xFFFFFFFF);
    }
}
