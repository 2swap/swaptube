#include "BezierStateCurve.h"

BezierStateCurve::BezierStateCurve(vector<StateSet> waypoints) {
    if(waypoints.size() < 2) throw runtime_error("Bezier waypoint list too small");

    unordered_set<string> keyset;
    for(const pair<string, string>& p : waypoints[0]) {
        keyset.insert(p.first);
    }

    for(int i = 1; i < waypoints.size(); i++) {
        unordered_set<string> keyset_test;
        for(const pair<string, string>& p : waypoints[i]) {
            keyset_test.insert(p.first);
        }
        if(keyset_test != keyset)
            throw runtime_error("Keysets don't match: waypoint 0 and waypoint " + to_string(i));
    }

    int sm1 = waypoints.size()-1;
    float tension = .25;
    string tension_str = to_string(tension);
    for(int i = 0; i < sm1; i++) {
        StateSet bezier;
        for(string key : keyset) {
            string before_point = waypoints[max(0,i-1)][key];
            string this_point = waypoints[i][key];
            string next_point = waypoints[i+1][key];
            string after_point = waypoints[min(i+2, sm1)][key];
            string control_point_1 = this_point + " " + next_point + " " + before_point + " - " + tension_str + " * +";
            string control_point_2 = next_point + " " + this_point + " " + after_point + " - " + tension_str + " * +";
            bezier[key] = this_point      + " " +
                          control_point_1 + " " +
                          control_point_2 + " " +
                          next_point      + " " +
                           + " {microblock_fraction} bezier";
        }
        entries.push_back(bezier);
    }
}

StateSet BezierStateCurve::pop_next_state_set() {
    if(entries.empty()) throw runtime_error("Attempted to pop non-existing entry from bezier curve");
    StateSet ret = entries.front();
    entries.pop_front();
    return ret;
}

int BezierStateCurve::size() const {
    return entries.size();
}
