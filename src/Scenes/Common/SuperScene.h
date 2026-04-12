#pragma once
#include "../Scene.h"
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <memory>
#include <string>

class SuperScene : public Scene {
public:
    void remove_subscene(const std::string& name);

    void fade_all_subscenes(const TransitionType tt, const double opacity);

    void fade_all_subscenes_except(const TransitionType tt, const std::string& name, const double opacity);

    void fade_subscene(const TransitionType tt, const std::string& name, const double opacity_final);

    void remove_all_subscenes();

    void remove_all_subscenes_except(const std::string& name);

    void move_to_front(const std::string& name);

    void move_to_back(const std::string& name);

    std::shared_ptr<Scene> get_subscene_pointer(const std::string& name);

protected:
    SuperScene(const vec2& dimensions = vec2(1, 1))
        : Scene(dimensions) {}

    bool needs_redraw() const override;

    void add_subscene_check_dupe(const std::string& name, std::shared_ptr<Scene> scene, bool behind = false);

    void on_end_transition_extra_behavior(const TransitionType tt) override;

    bool subscene_needs_redraw() const;

    const StateQuery populate_state_query() const override;

    std::list<std::string> render_order;
    std::unordered_map<std::string, std::shared_ptr<Scene>> subscenes;
};
