#include "CompositeScene.h"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <string>

CompositeScene::CompositeScene(const vec2& dimensions) : SuperScene(dimensions) {}

void CompositeScene::add_scene_fade_in(const TransitionType tt, std::shared_ptr<Scene> sc, std::string state_name, vec2 position, double opa, bool behind){
    add_scene(sc, state_name, position, behind);
    manager.set(state_name + ".opacity", "0");
    fade_subscene(tt, state_name, opa);
}

void CompositeScene::add_scene(std::shared_ptr<Scene> sc, const std::string& state_name, vec2 position, bool behind){
    manager.set({
        {state_name + ".x", std::to_string(position.x)},
        {state_name + ".y", std::to_string(position.y)},
        {state_name + ".angle", "0"},
    });
    add_subscene_check_dupe(state_name, sc, behind);
}

void CompositeScene::slide_subscene(const TransitionType tt, const std::string& name, const vec2 delta){
    manager.transition(tt, {
        {name + ".x", manager.get_equation_string(name + ".x") + " " + std::to_string(delta.x) + " +"},
        {name + ".y", manager.get_equation_string(name + ".y") + " " + std::to_string(delta.y) + " +"},
    });
}

void CompositeScene::draw() {
    const vec2& dimensions = get_dimensions();
    for (const std::string& name : render_order){
        const double opa = state[name + ".opacity"];
        if(opa < 0.001) continue;
        Pixels* p = nullptr;
        std::shared_ptr<Scene> subscene = subscenes[name];
        subscene->query(p);
        const vec2 center = dimensions*vec2(state[name + ".x"], state[name + ".y"]);

        // Angle in radians, clamped to [0, 2*pi)
        const float angle = extended_mod(state[name + ".angle"], 2*M_PI);

        const vec2 center_pix = center - subscene->get_dimensions()/2;

        if (angle > 0.01 && angle < 2*M_PI - 0.01) {
            cuda_overlay(
                pix.pixels.data(), pix.size,
                p->pixels.data(), p->size,
                center_pix, opa, angle
            );
        } else {
            cuda_overlay(
                pix.pixels.data(), pix.size,
                p->pixels.data(), p->size,
                center_pix, opa, 0
            );
        }
    }
}

const StateQuery CompositeScene::populate_state_query() const {
    StateQuery ret = SuperScene::populate_state_query();
    for (auto& subscene : subscenes){
        ret.insert(subscene.first + ".x");
        ret.insert(subscene.first + ".y");
        ret.insert(subscene.first + ".angle");
    };
    return ret;
}
