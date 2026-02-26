#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "DataObject.h"
using namespace std;

class GeometricItem {
public:
    string identifier;
    string label;
    bool use_state;
    bool old;
    bool draw_shape;
    GeometricItem(string id = "", string l = "", bool u_s = false, bool d_s = true) : identifier(id), label(l), use_state(u_s), old(false), draw_shape(d_s) {}
};

class GeometricPoint : public GeometricItem {
public:
    glm::vec2 position;
    float width_multiplier; // For rendering size
    GeometricPoint(glm::vec2 pos, string id = "", float wm = 1.0f, bool u_s = false, string l = "") : GeometricItem(id, l==""?id:l, u_s), position(pos), width_multiplier(wm) {}
};

class GeometricLine : public GeometricItem {
public:
    glm::vec2 start;
    glm::vec2 end;
    GeometricLine(glm::vec2 s, glm::vec2 e, string id = "", bool u_s = false, string l = "") : GeometricItem(id, l==""?id:l, u_s), start(s), end(e) {}
};

class GeometricConstruction : public DataObject {
public:
    void add(const GeometricPoint& p);
    void add(const GeometricLine& l);
    void clear();
    int size() const;

    void set_all_old();

    void remove_point(const string& id);
    void remove_line(const string& id);

    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
};
