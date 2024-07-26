#pragma once

using namespace std;

#include <list>
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>

class Object {
public:
    int color;
    float opacity;
    bool fixed;

    Object(int col, float op, bool fix) : color(col), opacity(op), fixed(fix) {}
};

class FixedObject : public Object {
public:
    const string dag_name;
    FixedObject(int col, float op, const string& dn)
        : Object(col, op, true), dag_name(dn) {}

    glm::vec3 get_position(const Dagger& dag) const {
        return glm::vec3(dag[dag_name + ".x"],
                         dag[dag_name + ".y"],
                         dag[dag_name + ".z"]);
    }
};

class MobileObject : public Object {
public:
    glm::vec3 position;
    glm::vec3 velocity;
    MobileObject(const glm::vec3& pos, int col, float op)
        : Object(col, op, false), position(pos), velocity(0) {}
};

class OrbitSim {
public:
    vector<FixedObject> fixed_objects;
    list<MobileObject> mobile_objects;
    bool mobile_interactions = true;

    void add_fixed_object(int color, float opacity, const string& dag_name) {
        fixed_objects.push_back(FixedObject(color, opacity, dag_name));
    }

    void add_mobile_object(const glm::vec3& position, int color, float opacity) {
        mobile_objects.push_back(MobileObject(position, color, opacity));
    }

    void iterate_physics(int multiplier, const Dagger& dag) {
        for (int step = 0; step < multiplier; ++step) iterate_physics_once(dag);
    }

    int predict_fate_of_object(glm::vec3 o, const Dagger& dag) {
        glm::vec3 velocity(0.f,0,0);
        float sqr_v = 0;

        float force_constant = dag["force_constant"];
        float collision_threshold_squared = square(dag["collision_threshold"]);
        float drag = dag["drag"];

        // Store positions in a vector
        std::vector<glm::vec3> positions;
        positions.reserve(fixed_objects.size());
        for (const auto& fixed_obj : fixed_objects) {
            positions.push_back(fixed_obj.get_position(dag));
        }

        while (true) {
            for (size_t i = 0; i < fixed_objects.size(); ++i) {
                glm::vec3 direction = positions[i] - o;
                float distance2 = glm::dot(direction, direction);
                if (distance2 < square(.1) && sqr_v < 1){
                    return fixed_objects[i].color;
                } else {
                    velocity += glm::normalize(direction) * magnitude_force_given_distance_squared(force_constant, distance2);
                }
            }

            velocity *= drag;
            o += velocity;
            sqr_v = 0;//glm::dot(velocity, velocity)/1000000;
        }
    }

private:
    void iterate_physics_once(const Dagger& dag) {
        float force_constant = dag["force_constant"];
        float collision_threshold_squared = square(dag["collision_threshold"]);
        float drag = dag["drag"];
        // Interactions between fixed objects and mobile objects
        for (auto it = mobile_objects.begin(); it != mobile_objects.end(); /*it is incremented elsewhere*/) {
            auto& obj1 = *it;
            bool deleted = false;

            for (const auto& fixed_obj : fixed_objects) {
                glm::vec3 direction = fixed_obj.get_position(dag) - obj1.position;
                float distance2 = glm::dot(direction, direction);
                if (distance2 < collision_threshold_squared) {
                    it = mobile_objects.erase(it);
                    deleted = true;
                    break;
                } else {
                    glm::vec3 acceleration = glm::normalize(direction) * magnitude_force_given_distance_squared(force_constant, distance2);
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
                            glm::vec3 acceleration = glm::normalize(direction) * magnitude_force_given_distance_squared(force_constant, distance2);
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
            object.position += object.velocity;
        }
    }

    inline float magnitude_force_given_distance_squared(float force_constant, float d2){
        return force_constant/(.1+d2);
    }
};

