#include "BouncingBallScene.h"

BouncingBallScene::BouncingBallScene(
    const int n,
    const double simulation_width,
    const double simulation_height,
    const double width,
    const double height
)
    : CoordinateScene(width, height), balls(n, simulation_width, simulation_height, .05f) {}

void BouncingBallScene::mark_data_unchanged() { balls.mark_unchanged(); }
void BouncingBallScene::change_data() { balls.iterate(); }
bool BouncingBallScene::check_if_data_changed() const { return balls.has_been_updated_since_last_scene_query(); }

void BouncingBallScene::draw(){
    for(const Ball& ball : balls.balls){
        vec2 pixel = point_to_pixel({ball.x, ball.y});
        vec2 rim = point_to_pixel({ball.x + balls.ball_radius, ball.y});
        pix.fill_circle(pixel.x, pixel.y, rim.x - pixel.x, 0xFFFFFFFF);
    }
}
