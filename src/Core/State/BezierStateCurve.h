#pragma once
#include <vector>
#include <list>
#include "StateManager.h"

class BezierStateCurve {
public:
    BezierStateCurve(vector<StateSet>);
    StateSet pop_next_state_set();
    int size() const;
private:
    list<StateSet> entries;
};
