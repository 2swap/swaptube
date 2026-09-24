#pragma once

#include <vector>
#include <string>
#include "../Host_Device_Shared/vec.h"
using namespace std;

class GeometricItem {
public:
    string identifier;
    string label;
    bool use_state;
    bool old;
    bool draw_shape;
    bool dying = false;
    GeometricItem(string id = "", string l = "", bool u_s = false, bool d_s = true) : identifier(id), label(l), use_state(u_s), old(false), draw_shape(d_s) {}
};

class GeometricPoint : public GeometricItem {
public:
    vec2 position;
    float width_multiplier;
    GeometricPoint(vec2 pos, string id = "", float wm = 1.0f, bool u_s = false, string l = "") : GeometricItem(id, l, u_s), position(pos), width_multiplier(wm) {}
};

class GeometricLine : public GeometricItem {
public:
    vec2 start;
    vec2 end;
    GeometricLine(vec2 s, vec2 e, string id = "", bool u_s = false, string l = "") : GeometricItem(id, l, u_s), start(s), end(e) {}
};

class GeometricConstruction {
public:
    void add(const GeometricPoint& p);
    void add(const GeometricLine& l);
    void clear();
    int size() const;

    void set_all_old();

    void remove_point(const string& id); // instant
    void remove_line(const string& id);  // instant

    void fade_point(const string& id);   // animate out over the next microblock, then prune
    void fade_line(const string& id);
    void prune_dead();                   // erase everything currently marked dying

    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
};
