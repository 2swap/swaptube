#pragma once

#include "../Scene.cpp"
#include <unordered_map>

class SuperScene : public Scene {
public:
    void remove_subscene(const string& name) {
        auto it = subscenes.find(name);
        if(it != subscenes.end()){
            it->second->state_manager.set_parent(nullptr);
            subscenes.erase(it);
        }
    }

    void fade_out_all_subscenes(bool micro = true) {
        for (auto& kv : subscenes) {
            unordered_map<string, string> map = {
                {kv.first + ".opacity", "0"}
            };
            if(micro) state_manager.microblock_transition(map);
            else      state_manager.macroblock_transition(map);
        }
    }

    void remove_all_subscenes() {
        for (auto& kv : subscenes){
            kv.second->state_manager.set_parent(nullptr);
        }
        subscenes.clear();
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
        scene->state_manager.set_parent(&state_manager);
        subscenes[name] = scene;
        if(name == "...fff.....cbba..cdda..e..a..e..hhhe") cout << "ASCD" << endl;
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
            if(kv.first == "...fff.....cbba..cdda..e..a..e..hhhe") cout << "SNR" << endl;
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
            if(kv.first == "...fff.....cbba..cdda..e..a..e..hhhe") cout << "PSQ" << endl;
            ret.insert(kv.first + ".opacity");
        };
        return ret;
    }

    unordered_map<string, shared_ptr<Scene>> subscenes;
};
