#pragma once

using namespace std;

#include <vector>
#include <glm/vec3.hpp> // Assuming you are using glm for vector operations

class Object {
public:
    float mass;
    int color;
    float opacity;
    bool fixed;

    Object(float m, int col, float op, bool fix) : mass(m), color(col), opacity(op), fixed(fix) {}
};

class FixedObject : public Object {
public:
    const string dag_name;
    FixedObject(float m, int col, float op, const string& dn)
        : Object(m, col, op, true), dag_name(dn) {}

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
    MobileObject(const glm::vec3& pos, float m, int col, float op)
        : Object(m, col, op, false), position(pos), velocity(0) {}
};

class OrbitSim {
public:
    vector<FixedObject> fixed_objects;
    vector<MobileObject> mobile_objects;
    bool mobile_interactions = true;

    void add_fixed_object(float mass, int color, float opacity, const string& dag_name) {
        fixed_objects.push_back(FixedObject(mass, color, opacity, dag_name));
    }

    void add_mobile_object(const glm::vec3& position, float mass, int color, float opacity) {
        mobile_objects.push_back(MobileObject(position, mass, color, opacity));
    }

    void iterate_physics(int multiplier, const Dagger& dag) {
        for (int step = 0; step < multiplier; ++step) iterate_physics_once(dag);
    }

    void iterate_physics_once(const Dagger& dag) {
        float force_constant = 0.00001f;

        // Interactions between fixed objects and mobile objects
        for (size_t i = 0; i < mobile_objects.size(); ++i) {
            glm::vec3 force(0.0f);
            for (size_t j = 0; j < fixed_objects.size(); ++j) {
                glm::vec3 direction = fixed_objects[j].get_position(dag) - mobile_objects[i].position;
                float distance = glm::length(direction);
                if (distance > 0) {
                    float force_magnitude = force_constant * mobile_objects[i].mass * fixed_objects[j].mass / distance;
                    glm::vec3 acceleration = glm::normalize(direction) * force_magnitude / mobile_objects[i].mass;
                    mobile_objects[i].velocity += acceleration;
                }
            }
        }

        if(mobile_interactions){
            // Interactions between two mobile objects
            for (size_t i = 0; i < mobile_objects.size(); ++i) {
                for (size_t j = i + 1; j < mobile_objects.size(); ++j) {
                    glm::vec3 direction = mobile_objects[j].position - mobile_objects[i].position;
                    float distance = glm::length(direction);
                    if (distance > 0) {
                        float force_magnitude = force_constant * mobile_objects[i].mass * mobile_objects[j].mass / distance;
                        glm::vec3 acceleration = glm::normalize(direction) * force_magnitude / mobile_objects[i].mass;
                        glm::vec3 acceleration2 = -glm::normalize(direction) * force_magnitude / mobile_objects[j].mass;
                        mobile_objects[i].velocity += acceleration;
                        mobile_objects[j].velocity += acceleration2;
                    }
                }
            }
        }

        for (auto& object : mobile_objects) {
            object.velocity *= 0.9999f;
            object.position += object.velocity;
        }
    }
};
