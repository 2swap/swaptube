#include "CompositeScene.h"
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <string>

extern "C" void cuda_overlay(
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle);

CompositeScene::CompositeScene(const vec2& dimensions) : SuperScene(dimensions) {}

void CompositeScene::add_scene_fade_in(const TransitionType tt, std::shared_ptr<Scene> sc, const std::string& state_name, const vec2& pos, double opa, bool behind){
    add_scene(sc, state_name, pos, behind);
    manager.set(state_name + ".opacity", "0");
    fade_subscene(tt, state_name, opa);
}

void CompositeScene::add_scene(std::shared_ptr<Scene> sc, const std::string& state_name, const vec2& pos, bool behind){
    manager.set({
        {state_name + ".x", std::to_string(pos.x)},
        {state_name + ".y", std::to_string(pos.y)},
        {state_name + ".angle", "0"},
    });
    add_subscene_check_dupe(state_name, sc, behind);
}

void CompositeScene::slide_subscene(const TransitionType tt, const std::string& name, const vec2& delta){
    manager.transition(tt, {
        {name + ".x", manager.get_equation_string(name + ".x") + " " + std::to_string(delta.x) + " +"},
        {name + ".y", manager.get_equation_string(name + ".y") + " " + std::to_string(delta.y) + " +"},
    });
}

void CompositeScene::draw() {
    ivec2 wh = get_width_height();
    for (const std::string& name : render_order){
        double opa = state[name + ".opacity"];
        if(opa < 0.001) continue;
        Pixels* p = nullptr;
        std::shared_ptr<Scene> subscene = subscenes[name];
        uint32_t* subscene_gpu_pix = subscene->query();
        const vec2 xy(wh * vec2(state[name + ".x"], state[name + ".y"]));

        const vec2 center = xy - subscene->get_width_height()/2;

        cuda_overlay(
            gpu_pix->get_ptr(), wh,
            subscene_gpu_pix, subscene->get_width_height(),
            center, opa, state[name + ".angle"]
        );
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
