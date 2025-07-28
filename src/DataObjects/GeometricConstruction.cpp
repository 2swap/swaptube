#pragma once
#include <vector>
#include "DataObject.cpp"

struct GeometricPoint {
    glm::vec2 position;
    GeometricPoint(glm::vec2 pos) : position(pos) {}
};
struct GeometricLine {
    glm::vec2 start;
    glm::vec2 end;
    GeometricLine(glm::vec2 s, glm::vec2 e) : start(s), end(e) {}
};
struct GeometricArc {
    glm::vec2 center;
    double start_angle;
    double end_angle;
    double radius;
    GeometricArc(glm::vec2 c, double sa, double ea, double r)
        : center(c), start_angle(sa), end_angle(ea), radius(r) {}
};

class GeometricConstruction : public DataObject {
public:
    void add_point(const GeometricPoint& p){
        points.push_back(p);
        mark_updated();
    }
    void add_line(const GeometricLine& l){
        lines.push_back(l);
        mark_updated();
    }
    void add_arc(const GeometricArc& a){
        arcs.push_back(a);
        mark_updated();
    }
    void clear() {
        points.clear();
        lines.clear();
        arcs.clear();
        mark_updated();
    }
    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
    vector<GeometricArc> arcs;
};
