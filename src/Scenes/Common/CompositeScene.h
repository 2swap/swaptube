#pragma once
#include "SuperScene.h"
#include <memory>
#include <string>

class CompositeScene : public SuperScene {
public:
    CompositeScene(const double width = 1, const double height = 1);

    // TODO glm vec2s for the positions for easier type checking in the arg list
    void add_scene_fade_in(const TransitionType tt, std::shared_ptr<Scene> sc, std::string state_name, double x = 0.5, double y = 0.5, double opa=1, bool behind = false);

    void add_scene(std::shared_ptr<Scene> sc, const std::string& state_name, double x = 0.5, double y = 0.5, bool behind = false);

    void slide_subscene(const TransitionType tt, const std::string& name, const double dx, const double dy);

    void draw() override;

    const StateQuery populate_state_query() const override;
};
