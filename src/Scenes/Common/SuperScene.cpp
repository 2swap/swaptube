#include "SuperScene.h"
#include <stdexcept>
#include <unordered_set>
#include <string>

void SuperScene::remove_subscene(const std::string& name) {
    auto it = subscenes.find(name);
    if(it != subscenes.end()){
        it->second->manager.set_parent(nullptr, name);
        render_order.remove(name); // Note this is inefficient but we rarely have more than 100 subscenes
        subscenes.erase(it);
    }
}

void SuperScene::fade_all_subscenes(const TransitionType tt, const double opacity) {
    for (auto& name : render_order) {
        fade_subscene(tt, name, opacity);
    }
}

void SuperScene::fade_all_subscenes_except(const TransitionType tt, const std::string& name, const double opacity) {
    for (auto& n : render_order) {
        if(n != name) fade_subscene(tt, n, opacity);
    }
}

void SuperScene::fade_subscene(const TransitionType tt, const std::string& name, const double opacity_final) {
    auto it = subscenes.find(name);
    if(it != subscenes.end())
        manager.transition(tt, {{name + ".opacity", std::to_string(opacity_final)}});
}

void SuperScene::remove_all_subscenes() {
    for (auto& name : render_order){
        subscenes[name]->manager.set_parent(nullptr, name);
    }
    subscenes.clear();
    render_order.clear();
}

void SuperScene::remove_all_subscenes_except(const std::string& name) {
    std::unordered_set<std::string> to_remove;
    for (auto& n : render_order){
        if(n != name) to_remove.insert(n);
    }
    for(const std::string& s : to_remove) remove_subscene(s);
}

void SuperScene::move_to_front(const std::string& name) {
    render_order.remove(name);
    render_order.push_back(name);
}

void SuperScene::move_to_back(const std::string& name) {
    render_order.remove(name);
    render_order.push_front(name);
}

std::shared_ptr<Scene> SuperScene::get_subscene_pointer(const std::string& name) {
    auto it = subscenes.find(name);
    if(it != subscenes.end())
        return it->second;
    else
        throw std::runtime_error("Error: Attempted to get pointer to non-existent subscene: " + name);
}

void SuperScene::add_subscene_check_dupe(const std::string& name, std::shared_ptr<Scene> scene, bool behind) {
    if(!scene) throw std::runtime_error("Error: Attempted to add a null subscene to superscene: " + name);
    if(subscenes.find(name) != subscenes.end()) throw std::runtime_error("Error: Added two subscenes of the same name to superscene: " + name);
    scene->manager.set_parent(&manager, name);
    subscenes[name] = scene;
    if(behind) render_order.push_front(name);
    else       render_order.push_back(name);
    manager.set(name + ".opacity", "1");
}

void SuperScene::change_data() {
    Scene::change_data();
    for(const auto& kv : subscenes) {
        kv.second->update();
    }
}

void SuperScene::on_end_transition_extra_behavior(const TransitionType tt) {
    for(const auto& kv : subscenes){
        kv.second->on_end_transition(tt);
    }
}
