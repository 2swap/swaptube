#pragma once

#include "../Common/CoordinateScene.cpp"

class PendulumPointsScene : public CoordinateScene {
public:
    PendulumPointsScene(const PendulumGrid& pg, const double width = 1, const double height = 1)
        : CoordinateScene(width, height), grid(pg) {
        state_manager.add_equation("points_opacity", "1");
    }

    void draw() override {
        CoordinateScene::draw();
        render_points();
    }

    void render_points() {
        double points_opacity = state["points_opacity"];
        for (int y = 0; y < grid.h; ++y) {
            for (int x = 0; x < grid.w; ++x) {
                int i = x + y * grid.w;
                PendulumState this_ps = grid.pendulum_states[i];
                PendulumState start_ps = grid.start_states[i];
                int point_color = pendulum_color(start_ps.theta1, start_ps.theta2);
                const pair<double, double> curr_point = make_pair(this_ps.theta1, this_ps.theta2);
                const pair<int, int> curr_pixel = point_to_pixel(curr_point);
                pix.fill_circle(curr_pixel.first, curr_pixel.second, 2, colorlerp(TRANSPARENT_BLACK, point_color, points_opacity));
            }
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery s = CoordinateScene::populate_state_query();
        s.insert("points_opacity");
        s.insert("physics_multiplier");
        s.insert("rk4_step_size");
        return s;
    }
    void mark_data_unchanged() override { grid.mark_unchanged(); }
    void change_data() override { grid.iterate_physics(state["physics_multiplier"], state["rk4_step_size"]); }
    bool check_if_data_changed() const override { return grid.has_been_updated_since_last_scene_query(); }

private:
    PendulumGrid grid;
};
