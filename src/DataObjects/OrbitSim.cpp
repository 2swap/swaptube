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
    const string state_name;
    FixedObject(int col, const string& dn)
        : Object(col, true), state_name(dn) {}

    float get_opacity(const StateManager& state) const {
        return state[state_name + ".opacity"];
    }

    glm::dvec3 get_position(const StateManager& state) const {
        return glm::dvec3(state[state_name + ".x"],
                         state[state_name + ".y"],
                         state[state_name + ".z"]);
    }
};

class MobileObject : public Object {
public:
    glm::dvec3 position;
    glm::dvec3 velocity;
    MobileObject(const glm::dvec3& pos, int col)
        : Object(col, false), position(pos), velocity(0) {}
};

float global_force_constant = 0.001;

class OrbitSim : public DataObject {
public:
    list<FixedObject> fixed_objects;
    list<MobileObject> mobile_objects;
    bool mobile_interactions = false;

    void remove_fixed_object(const string& state_name) {
        fixed_objects.remove_if([&state_name](const FixedObject& obj) { return obj.state_name == state_name; });
        mark_updated();
    }

    void add_fixed_object(int color, const string& state_name) {
        fixed_objects.push_back(FixedObject(color, state_name));
        mark_updated();
    }

    void add_mobile_object(const glm::dvec3& position, int color) {
        mobile_objects.push_back(MobileObject(position, color));
        mark_updated();
    }

    void iterate_physics(int multiplier, const StateManager& state) {
        for (int step = 0; step < multiplier; ++step) iterate_physics_once(state);
    }

    void get_parameters_from_state(float& tick_duration, float& collision_threshold_squared, float& drag, const StateManager& state) {
        tick_duration = state["tick_duration"];
        collision_threshold_squared = square(state["collision_threshold"]);
        drag = pow(state["drag"], tick_duration);
    }

    bool get_next_step(glm::dvec3& pos, glm::dvec3& vel, int& color, const StateManager& state) {
        float tick_duration, collision_threshold_squared, drag;
        get_parameters_from_state(tick_duration, collision_threshold_squared, drag, state);
        float eps = state["eps"];

        float v2 = glm::dot(vel, vel);
        for (const FixedObject& fo : fixed_objects) {
            glm::dvec3 direction = fo.get_position(state) - pos;
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

    int predict_fate_of_object(glm::dvec3 pos, const StateManager& state) {
        glm::dvec3 vel(0.f, 0, 0);

        int color = 0;
        for (int i = 0; i < 10000; ++i) 
            if(get_next_step(pos, vel, color, state))
                return color;
        return 0;
    }

    void get_fixed_object_data_for_cuda(vector<glm::dvec3>& positions, vector<int>& colors, vector<float>& opacities, const StateManager& state) {
        int num_positions = fixed_objects.size();
        positions.resize(num_positions);
           colors.resize(num_positions);
        opacities.resize(num_positions);
        int i = 0;
        for (const FixedObject& fo : fixed_objects) {
            positions[i] = fo.get_position(state);
               colors[i] = fo.color;
            opacities[i] = fo.get_opacity(state);
            i++;
        }
    }

private:
    void iterate_physics_once(const StateManager& state) {
        float tick_duration, collision_threshold_squared, drag;
        get_parameters_from_state(tick_duration, collision_threshold_squared, drag, state);
        float eps = state["eps"];
        // Interactions between fixed objects and mobile objects
        for (auto it = mobile_objects.begin(); it != mobile_objects.end(); /*it is incremented elsewhere*/) {
            auto& obj1 = *it;
            bool deleted = false;
            float v2 = glm::dot(obj1.velocity, obj1.velocity);

            for (const auto& fixed_obj : fixed_objects) {
                glm::dvec3 direction = fixed_obj.get_position(state) - obj1.position;
                float distance2 = glm::dot(direction, direction);
                if (distance2 < collision_threshold_squared && v2 < global_force_constant) {
                    it = mobile_objects.erase(it);
                    deleted = true;
                    break;
                } else {
                    glm::dvec3 acceleration = tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared(eps, distance2);
                    obj1.velocity += acceleration;
                }
            }

            if (!deleted) {
                if (mobile_interactions) {
                    // Interactions between two mobile objects
                    for (auto it2 = std::next(it); it2 != mobile_objects.end(); it2++) {
                        auto& obj2 = *it2;
                        glm::dvec3 direction = obj2.position - obj1.position;
                        float distance2 = glm::dot(direction, direction);
                        if (distance2 > 0) {
                            glm::dvec3 acceleration = tick_duration * glm::normalize(direction) * magnitude_force_given_distance_squared(eps, distance2);
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

