#pragma once
#include <vector>
#include <string>
#include "DataObject.cpp"

class GeometricItem {
public:
    string label;
    bool use_state;
    bool old;
    GeometricItem(string l = "", bool u_s = false) : label(l), use_state(u_s), old(false) {}
};

class GeometricPoint : public GeometricItem {
public:
    glm::vec2 position;
    float width_multiplier; // For rendering size
    GeometricPoint(glm::vec2 pos, string l = "", float wm = 1.0f, bool u_s = false) : GeometricItem(l, u_s), position(pos), width_multiplier(1.0f) {}
};

class GeometricLine : public GeometricItem {
public:
    glm::vec2 start;
    glm::vec2 end;
    GeometricLine(glm::vec2 s, glm::vec2 e, string l = "", bool u_s = false) : GeometricItem(l, u_s), start(s), end(e) {}
};

class GeometricArc : public GeometricItem {
public:
    glm::vec2 center;
    double start_angle;
    double end_angle;
    double radius;
    GeometricArc(glm::vec2 c, double sa, double ea, double r, string l = "", bool u_s = false) : GeometricItem(l, u_s), center(c), start_angle(sa), end_angle(ea), radius(r) {}
};

class GeometricConstruction : public DataObject {
public:
    void add(const GeometricPoint& p){
        points.push_back(p);
        mark_updated();
    }
    void add(const GeometricLine& l){
        lines.push_back(l);
        mark_updated();
    }
    void add(const GeometricArc& a){
        arcs.push_back(a);
        mark_updated();
    }
    void clear() {
        points.clear();
        lines.clear();
        arcs.clear();
        mark_updated();
    }
    int size() const {
        return points.size() + lines.size() + arcs.size();
    }
    void set_all_old() {
        for (auto& p : points) p.old = true;
        for (auto& l : lines) l.old = true;
        for (auto& a : arcs) a.old = true;
        mark_updated();
    }
    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
    vector<GeometricArc> arcs;
};
