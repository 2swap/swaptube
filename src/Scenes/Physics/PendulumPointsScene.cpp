#include "PendulumPointsScene.h"

PendulumPointsScene::PendulumPointsScene(const PendulumGrid& pg, const vec2& dimensions)
    : CoordinateScene(dimensions), grid(pg) {
    manager.set("points_opacity", "1");
}

void PendulumPointsScene::draw() {
    CoordinateScene::draw();
    render_points();
}

void PendulumPointsScene::render_points() {
    double points_opacity = state["points_opacity"];
    for (int y = 0; y < grid.h; ++y) {
        for (int x = 0; x < grid.w; ++x) {
            int i = x + y * grid.w;
            PendulumState this_ps = grid.pendulum_states[i];
            PendulumState start_ps = grid.start_states[i];
            int point_color = pendulum_color(start_ps.theta1, start_ps.theta2, start_ps.p1, start_ps.p2);
            vec2 curr_point = vec2(this_ps.theta1, this_ps.theta2);
            vec2 curr_pixel = point_to_pixel(curr_point);
            pix.fill_circle(
                curr_pixel.x,
                curr_pixel.y,
                get_geom_mean_size()/200,
                colorlerp(TRANSPARENT_BLACK, point_color, points_opacity)
            );
        }
    }
}

const StateQuery PendulumPointsScene::populate_state_query() const {
    StateQuery s = CoordinateScene::populate_state_query();
    state_query_insert_multiple(s, {"points_opacity", "physics_multiplier", "rk4_step_size"});
    return s;
}
