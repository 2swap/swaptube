#pragma once

#include "../Scene.h"
#include "../Common/CoordinateScene.h"
#include "../../DataObjects/Permutation.h"


class PermutationScene : public CoordinateScene {
    public:
        PermutationScene(const std::string file_name, const vec2& dimensions = vec2(1, 1));
        void draw() override;
        Permutation the_perm;
        void on_end_transition_extra_behavior(const TransitionType tt) override;
        void move(const std::string orbit_name);
        vec2 get_place_position_from_state(const string& place_name);
    private:
        std::string moving_orbit_name;
};
