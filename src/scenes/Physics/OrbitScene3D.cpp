#pragma once

#include "../Scene.cpp"
#include "../Common/3DScene.cpp"
#include "OrbitSim.cpp"

class OrbitScene : public ThreeDimensionScene {
public:
    OrbitScene(OrbitSim* sim, const int width = VIDEO_WIDTH, const int height = VIDEO_HEIGHT)
        : ThreeDimensionScene(width, height), simulation(sim) {}

    void sim_to_3d() {
        points.clear();

        for (auto it = lines.begin(); it != lines.end(); ) {
            it->opacity *= 0.98f;
            if (it->opacity < 0.05f) {
                it = lines.erase(it); // Remove line and get the next iterator
            } else {
                ++it; // Move to the next line
            }
        }

        for (const auto& obj : simulation->mobile_objects) {
            auto position = obj.position;
            points.push_back(Point(obj.position, obj.color, NORMAL, obj.opacity));
            //lines.push_back(Line(position, obj.position, 0xff00ff00, 1));
        }
        for (const auto& obj : simulation->fixed_objects) {
            points.push_back(Point(obj.get_position(dag), obj.color, RING, obj.opacity));
        }
    }

    void draw(Pixels*& p) override {
        simulation->iterate_physics(physics_multiplier, dag);
        sim_to_3d();
        ThreeDimensionScene::draw();
    }

    int physics_multiplier = 1;

protected:
    OrbitSim* simulation;
};
