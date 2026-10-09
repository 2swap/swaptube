#pragma once
#include <vector>
#include <list>
#include "StateManager.h"
#include "../../Scenes/Scene.h"

class BezierStateCurve {
public:
    BezierStateCurve(vector<StateSet>);
    StateSet pop_next_state_set();
    int size() const;
    void run_curve(Scene&);
private:
    vector<StateSet> _waypoints;
    list<StateSet> entries;
};
