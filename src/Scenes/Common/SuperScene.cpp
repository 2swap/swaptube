#pragma once

#include "../Scene.cpp"
#include <unordered_map>
#include <list>

class SuperScene : public Scene {
public:
    void remove_subscene(const string& name) {
        auto it = subscenes.find(name);
        if(it != subscenes.end()){
            it->second->manager.set_parent(nullptr, name);
            render_order.remove(name); // Note this is inefficient but we rarely have more than 100 subscenes
            subscenes.erase(it);
        }
    }

    void fade_all_subscenes(const TransitionType tt, const double opacity) {
        for (auto& name : render_order) {
            fade_subscene(tt, name, opacity);
        }
    }

    void fade_all_subscenes_except(const TransitionType tt, const string& name, const double opacity) {
        for (auto& n : render_order) {
            if(n != name) fade_subscene(tt, n, opacity);
        }
    }

    void fade_subscene(const TransitionType tt, const string& name, const double opacity_final) {
        auto it = subscenes.find(name);
        if(it != subscenes.end())
            manager.transition(tt, {{name + ".opacity", to_string(opacity_final)}});
    }

    void remove_all_subscenes() {
        for (auto& name : render_order){
            subscenes[name]->manager.set_parent(nullptr, name);
        }
        subscenes.clear();
        render_order.clear();
    }

    void remove_all_subscenes_except(const string& name) {
        unordered_set<string> to_remove;
        for (auto& n : render_order){
            if(n != name) to_remove.insert(n);
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

    void add_subscene_check_dupe(const string& name, shared_ptr<Scene> scene, bool behind = false) {
        if(!scene) throw runtime_error("Error: Attempted to add a null subscene to superscene: " + name);
        if(subscenes.find(name) != subscenes.end()) throw runtime_error("Error: Added two subscenes of the same name to superscene: " + name);
        scene->manager.set_parent(&manager, name);
        subscenes[name] = scene;
        if(behind) render_order.push_front(name);
        else       render_order.push_back(name);
        manager.set(name + ".opacity", "1");
    }

    void change_data() override {
        for(const auto& kv : subscenes) kv.second->update();
    }

    bool check_if_data_changed() const override {
        for(const auto& kv : subscenes){
            if(kv.second->check_if_data_changed()) return true;
        }
        return false;
    }

    void on_end_transition_extra_behavior(const TransitionType tt) override {
        for(const auto& kv : subscenes){
            kv.second->on_end_transition(tt);
        }
    }

    bool subscene_needs_redraw() const {
        for (const auto& name : render_order){
            if(state[name + ".opacity"] > 0.01 && subscenes.at(name)->needs_redraw()) return true;
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
        for (const auto& name : render_order){
            ret.insert(name + ".opacity");
        };
        return ret;
    }

    list<string> render_order;
    unordered_map<string, shared_ptr<Scene>> subscenes;
};
