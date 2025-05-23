#pragma once

#include "../Scene.cpp"
#include <unordered_map>

class SuperScene : public Scene {
public:
    void remove_subscene(const string& name) {
        auto it = subscenes.find(name);
        if(it != subscenes.end()){
            it->second->state_manager.set_parent(nullptr, name);
            subscenes.erase(it);
        }
    }

    void fade_all_subscenes(double opacity, bool micro = true) {
        for (auto& kv : subscenes) {
            fade_subscene(kv.first, opacity, micro);
        }
    }

    void fade_all_subscenes_except(const string& name, double opacity, bool micro = true) {
        for (auto& kv : subscenes) {
            if(kv.first != name) fade_subscene(kv.first, opacity, micro);
        }
    }

    void fade_subscene(const string& name, double opacity, bool micro = true) {
        auto it = subscenes.find(name);
        if(it != subscenes.end()){
            unordered_map<string, string> map = {
                {name + ".opacity", to_string(opacity)}
            };
            if(micro) state_manager.microblock_transition(map);
            else      state_manager.macroblock_transition(map);
        }
    }

    void remove_all_subscenes() {
        for (auto& kv : subscenes){
            kv.second->state_manager.set_parent(nullptr, kv.first);
        }
        subscenes.clear();
    }

    void remove_all_subscenes_except(const string& name) {
        unordered_set<string> to_remove;
        for (auto& kv : subscenes){
            if(kv.first != name) to_remove.insert(kv.first);
        }
        for(const string& s : to_remove) remove_subscene(s);
    }

protected:
    SuperScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    bool needs_redraw() const override {
        bool state_change = check_if_state_changed();
        bool data_change = check_if_data_changed();
        bool subscene_change = subscene_needs_redraw();
        return !has_ever_rendered || state_change || data_change || subscene_change;
    }

    void add_subscene_check_dupe(const string& name, shared_ptr<Scene> scene) {
        if(subscenes.find(name) != subscenes.end())
            throw runtime_error("Error: Added two subscenes of the same name to superscene: " + name);
        scene->state_manager.set_parent(&state_manager, name);
        subscenes[name] = scene;
        state_manager.set({
            {name + ".opacity", "1"},
        });
    }

    void change_data() override {
        for(const auto& kv : subscenes){
            kv.second->update();
        }
    }

    bool check_if_data_changed() const override {
        for(const auto& kv : subscenes){
            if(kv.second->check_if_data_changed()) return true;
        }
        return false;
    }

    void on_end_transition_extra_behavior(bool is_macroblock) override {
        for(const auto& kv : subscenes){
            kv.second->on_end_transition(is_macroblock);
        }
    }

    bool subscene_needs_redraw() const {
        for (const auto& kv : subscenes){
            if(state[kv.first + ".opacity"] > 0.01 && kv.second->needs_redraw()) return true;
        }
        return false;
    }

    void mark_data_unchanged() override {
        for(const auto& kv : subscenes){
            kv.second->mark_data_unchanged();
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery ret;
        for (const auto& kv : subscenes){
            ret.insert(kv.first + ".opacity");
        };
        return ret;
    }

    unordered_map<string, shared_ptr<Scene>> subscenes;
};
