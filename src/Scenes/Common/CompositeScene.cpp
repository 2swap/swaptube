#include "CompositeScene.h"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <string>

CompositeScene::CompositeScene(const double width, const double height)
    : SuperScene(width, height) {}

void CompositeScene::add_scene_fade_in(const TransitionType tt, std::shared_ptr<Scene> sc, std::string state_name, double x, double y, double opa, bool behind){
    add_scene(sc, state_name, x, y, behind);
    manager.set(state_name + ".opacity", "0");
    fade_subscene(tt, state_name, opa);
}

void CompositeScene::add_scene(std::shared_ptr<Scene> sc, const std::string& state_name, double x, double y, bool behind){
    manager.set({
        {state_name + ".x", std::to_string(x)},
        {state_name + ".y", std::to_string(y)},
        {state_name + ".angle", "0"},
    });
    add_subscene_check_dupe(state_name, sc, behind);
}

void CompositeScene::slide_subscene(const TransitionType tt, const std::string& name, const double dx, const double dy){
    manager.transition(tt, {
        {name + ".x", manager.get_equation_string(name + ".x") + " " + std::to_string(dx) + " +"},
        {name + ".y", manager.get_equation_string(name + ".y") + " " + std::to_string(dy) + " +"},
    });
}

void CompositeScene::draw() {
    int w = get_width();
    int h = get_height();
    for (const std::string& name : render_order){
        double opa = state[name + ".opacity"];
        if(opa < 0.01) continue;
        Pixels* p = nullptr;
        std::shared_ptr<Scene> subscene = subscenes[name];
        subscene->query(p);
        int x = w*state[name + ".x"];
        int y = h*state[name + ".y"];

        // Angle in radians, clamped to [0, 2*pi)
        float angle = extended_mod(state[name + ".angle"], 2*M_PI);

        float center_x = x - subscene->get_width ()/2;
        float center_y = y - subscene->get_height()/2;

        if (angle > 0.01 && angle < 2*M_PI - 0.01) {
            cuda_overlay_with_rotation(
                pix.pixels.data(), pix.w, pix.h,
                pix.pixels.data(), pix.w, pix.h,
                center_x, center_y, opa, angle
            );
        }
        else {
            cuda_overlay(
                pix.pixels.data(), pix.w, pix.h,
                p->pixels.data(), p->w, p->h,
                center_x, center_y, opa
            );
        }
    }
}

const StateQuery CompositeScene::populate_state_query() const {
    StateQuery ret = SuperScene::populate_state_query();
    for (auto& subscene : subscenes){
        ret.insert(subscene.first + ".x");
        ret.insert(subscene.first + ".y");
    };
    return ret;
}
