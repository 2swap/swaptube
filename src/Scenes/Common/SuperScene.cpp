#pragma once

#include "../Scene.cpp"

struct NamedSubscene {
    Scene* ptr;
    string state_manager_name;
};

class SuperScene : public Scene {
public:
    SuperScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    bool needs_redraw() const override {
        bool state_change = check_if_state_changed();
        bool data_change = check_if_data_changed();
        bool subscene_change = subscene_needs_redraw();
        return !has_ever_rendered || state_change || data_change || subscene_change;
    }

    void change_data() override {
        for(const auto& subscene : subscenes){
            subscene.ptr->update();
        }
    }

    bool check_if_data_changed() const override {
        for(const auto& subscene : subscenes){
            if(subscene.ptr->check_if_data_changed()) return true;
        }
        return false;
    }

    void on_end_transition(bool is_macroblock) override {
        for(const auto& subscene : subscenes){
            subscene.ptr->on_end_transition(is_macroblock);
            subscene.ptr->state_manager.close_microblock_transitions();
            if(is_macroblock) subscene.ptr->state_manager.close_macroblock_transitions();
        }
    }

    bool subscene_needs_redraw() const {
        for (auto& subscene : subscenes){
            if(state[subscene.state_manager_name + ".opacity"] > 0.01 && subscene.ptr->needs_redraw()) return true;
        }
        return false;
    }

    void fade_out_all_scenes() {
        for (auto& subscene : subscenes) {
            unordered_map<string, string> map = {
                {subscene.state_manager_name + ".opacity", "0"}
            };
            state_manager.microblock_transition(map);
        }
    }

    void remove_all_scenes() {
        for (auto& subscene : subscenes){
            subscene.ptr->state_manager.set_parent(nullptr);
        }
        subscenes.clear();
    }

    void mark_data_unchanged() override {
        for(const auto& subscene : subscenes){
            subscene.ptr->mark_data_unchanged();
        }
    }

    void remove_scene(Scene* sc) {
        sc->state_manager.set_parent(nullptr);
        subscenes.erase(std::remove_if(subscenes.begin(), subscenes.end(),
                                    [sc](const NamedSubscene& subscene) {
                                        return subscene.ptr == sc;
                                    }), subscenes.end());
    }
protected:
    vector<NamedSubscene> subscenes;
};
