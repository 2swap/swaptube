#include "GeometricConstruction.h"
#include <algorithm>

void GeometricConstruction::add(const GeometricPoint& p){
    points.push_back(p);
    mark_updated();
}
void GeometricConstruction::add(const GeometricLine& l){
    lines.push_back(l);
    mark_updated();
}
void GeometricConstruction::clear() {
    points.clear();
    lines.clear();
    mark_updated();
}
int GeometricConstruction::size() const {
    return points.size() + lines.size();
}

void GeometricConstruction::set_all_old() {
    for (auto& p : points) p.old = true;
    for (auto& l : lines) l.old = true;
    mark_updated();
}

void GeometricConstruction::remove_point(const string& id) {
    points.erase(std::remove_if(points.begin(), points.end(), [&](const GeometricPoint& p){
        return p.identifier == id;
    }), points.end());
    mark_updated();
}

void GeometricConstruction::remove_line(const string& id) {
    lines.erase(std::remove_if(lines.begin(), lines.end(), [&](const GeometricLine& l){
        return l.identifier == id;
    }), lines.end());
    mark_updated();
}
