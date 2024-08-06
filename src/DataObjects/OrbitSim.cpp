#pragma once

using namespace std;

#include <list>
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>
#include "DataObject.cpp"

class Object {
public:
    int color;
    bool fixed;

    Object(int col, bool fix) : color(col), fixed(fix) {}
};

class FixedObject : public Object {
public:
    const string state_manager_name;
    FixedObject(int col, const string& dn)
        : Object(col, true), state_manager_name(dn) {}

    float get_opacity(const StateManager& state_manager const {
        return state_manager[state_manager_name + ".opacity"];
    }

    glm::vec3 get_position(const StateManager& state_manager const {
        return glm::vec3(state_manager[state_manager_name + ".x"],
                         state_manager[state_manager_name + ".y"],
                         state_manager[state_manager_name + ".z"]);
    }
};

class MobileObject : public Object {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    MobileObject(const glm::vec3& pos, int col)
        : Object(col, false), position(pos), velocity(0) {}
};

float global_force_constant = 0.001;

class OrbitSim : public DataObject {
public:
    list<FixedObject> fixed_objects;
    list<MobileObject> mobile_objects;
    bool mobile_interactions = false;

    void remove_fixed_object(const string& state_manager_name) {
        fixed_objects.remove_if([&state_manager_name](const FixedObject& obj) { return obj.state_manager_name == state_manager_name; });
        mark_updated();
    }

    void add_fixed_object(int color, const string& state_manager_name) {
        fixed_objects.push_back(FixedObject(color, state_manager_name));
        mark_updated();
    }

    void add_mobile_object(const glm::vec3& position, int color) {
        mobile_objects.push_back(MobileObject(position, color));
        mark_updated();
    }

    void iterate_physics(int multiplier, const StateManager& state_manager {
        for (int step = 0; step < multiplier; ++step) iterate_physics_oncestate_manager;
    }

    void get_parameters_from_state_manager(float& tick_duration, float& collision_threshold_squared, float& drag, const StateManager& state_manager{
        tick_duration = state_manager["tick_duration"];
        collision_threshold_squared = square(state_manager["collision_threshold"]);
        drag = pow(state_manager["drag"], tick_duration);
    }

    bool get_next_step(glm::vec3& pos, glm::vec3& vel, int& color, const StateManager& state_manager{
        float tick_duration, collision_threshold_squared, drag;
        get_parameters_from_state_manager(tick_duration, collision_threshold_squared, drag, state_manager;
        float eps = state_manager["eps"];

        float v2 = glm::dot(vel, vel);
        for (const FixedObject& fo : fixed_objects) {
            glm::vec3 direction = fo.get_positionstate_manager - pos;
            float distance2 = glm::dot(direction, direction);
            if (distance2 < collision_threshold_squared && v2 < global_force_constant) {
                color = fo.color;
                return true;
            } else {
                vel += tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared(eps, distance2);
            }
        }

        vel *= drag;
        pos += vel * tick_duration;
        return false;
    }

    int predict_fate_of_object(glm::vec3 pos, const StateManager& state_manager {
        glm::vec3 vel(0.f, 0, 0);

        int color = 0;
        for (int i = 0; i < 10000; ++i) 
            if(get_next_step(pos, vel, color, state_manager)
                return color;
        return 0;
    }

    void get_fixed_object_data_for_cuda(vector<glm::vec3>& positions, vector<int>& colors, vector<float>& opacities, const StateManager& state_manager{
        int num_positions = fixed_objects.size();
        positions.resize(num_positions);
           colors.resize(num_positions);
        opacities.resize(num_positions);
        int i = 0;
        for (const FixedObject& fo : fixed_objects) {
            positions[i] = fo.get_positionstate_manager;
               colors[i] = fo.color;
            opacities[i] = fo.get_opacitystate_manager;
            i++;
        }
    }

private:
    void iterate_physics_once(const StateManager& state_manager {
        float tick_duration, collision_threshold_squared, drag;
        get_parameters_from_state_manager(tick_duration, collision_threshold_squared, drag, state_manager;
        float eps = state_manager["eps"];
        // Interactions between fixed objects and mobile objects
        for (auto it = mobile_objects.begin(); it != mobile_objects.end(); /*it is incremented elsewhere*/) {
            auto& obj1 = *it;
            bool deleted = false;
            float v2 = glm::dot(obj1.velocity, obj1.velocity);

            for (const auto& fixed_obj : fixed_objects) {
                glm::vec3 direction = fixed_obj.get_positionstate_manager - obj1.position;
                float distance2 = glm::dot(direction, direction);
                if (distance2 < collision_threshold_squared && v2 < global_force_constant) {
                    it = mobile_objects.erase(it);
                    deleted = true;
                    break;
                } else {
                    glm::vec3 acceleration = tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared(eps, distance2);
                    obj1.velocity += acceleration;
                }
            }

            if (!deleted) {
                if (mobile_interactions) {
                    // Interactions between two mobile objects
                    for (auto it2 = std::next(it); it2 != mobile_objects.end(); it2++) {
                        auto& obj2 = *it2;
                        glm::vec3 direction = obj2.position - obj1.position;
                        float distance2 = glm::dot(direction, direction);
                        if (distance2 > 0) {
                            glm::vec3 acceleration = tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared(eps, distance2);
                            obj1.velocity += acceleration;
                            obj2.velocity -= acceleration;
                        }
                    }
                }
                it++;
            }
        }

        for (auto& object : mobile_objects){
            object.velocity *= drag;
            object.position += object.velocity * tick_duration;
            if(glm::dot(object.velocity, object.velocity) > 0.00001) mark_updated();
        }
    }

    inline float magnitude_force_given_distance_squared(float eps, float d2){
        return global_force_constant/(eps+d2);
    }
};

