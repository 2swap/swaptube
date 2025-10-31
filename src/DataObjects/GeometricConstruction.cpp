#pragma once
#include <vector>
#include <string>
#include "DataObject.cpp"

class GeometricItem {
public:
    string identifier;
    string label;
    bool use_state;
    bool old;
    GeometricItem(string id = "", string l = "", bool u_s = false) : identifier(id), label(l), use_state(u_s), old(false) {}
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
    void add(const GeometricPoint& p){
        points.push_back(p);
        mark_updated();
    }
    void add(const GeometricLine& l){
        lines.push_back(l);
        mark_updated();
    }
    void clear() {
        points.clear();
        lines.clear();
        mark_updated();
    }
    int size() const {
        return points.size() + lines.size();
    }
    void set_all_old() {
        for (auto& p : points) p.old = true;
        for (auto& l : lines) l.old = true;
        mark_updated();
    }
    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
};
