#pragma once
#include <vector>
#include "DataObject.cpp"

class GeometricPoint {
public:
    glm::vec2 position;
};
class GeometricLine {
public:
    glm::vec2 start;
    glm::vec2 end;
};
class GeometricArc {
public:
    glm::vec2 center;
    double start_angle;
    double end_angle;
    double radius;
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
    vector<GeometricPoint> points;
    vector<GeometricLine> lines;
    vector<GeometricArc> arcs;
};
