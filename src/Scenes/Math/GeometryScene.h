#pragma once

#include "../Scene.h"
#include "../../DataObjects/GeometricConstruction.h"
#include "../Common/CoordinateScene.h"
#include <string>
#include <sstream>
#include <unordered_set>
#include <list>
#include <utility>

class GeometryScene : public CoordinateScene {
public:
    GeometricConstruction construction;
    GeometryScene(const vec2& dimensions = vec2(1, 1));

    void draw() override;

    const StateQuery populate_state_query() const override;

    void on_end_transition_extra_behavior(const TransitionType tt) override;
};
