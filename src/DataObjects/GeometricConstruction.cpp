#include "GeometricConstruction.h"
#include <algorithm>

void GeometricConstruction::add(const GeometricPoint& p){
    points.push_back(p);
}
void GeometricConstruction::add(const GeometricLine& l){
    lines.push_back(l);
}
void GeometricConstruction::clear() {
    points.clear();
    lines.clear();
}
int GeometricConstruction::size() const {
    return points.size() + lines.size();
}

void GeometricConstruction::set_all_old() {
    for (auto& p : points) p.old = true;
    for (auto& l : lines) l.old = true;
}

void GeometricConstruction::remove_point(const string& id) {
    points.erase(std::remove_if(points.begin(), points.end(), [&](const GeometricPoint& p){
        return p.identifier == id;
    }), points.end());
}

void GeometricConstruction::remove_line(const string& id) {
    lines.erase(std::remove_if(lines.begin(), lines.end(), [&](const GeometricLine& l){
        return l.identifier == id;
    }), lines.end());
}

void GeometricConstruction::fade_point(const string& id) {
    for (auto& p : points) if (p.identifier == id) p.dying = true;
}

void GeometricConstruction::fade_line(const string& id) {
    for (auto& l : lines) if (l.identifier == id) l.dying = true;
}

void GeometricConstruction::prune_dead() {
    points.erase(std::remove_if(points.begin(), points.end(), [](const GeometricPoint& p){ return p.dying; }), points.end());
    lines.erase(std::remove_if(lines.begin(), lines.end(), [](const GeometricLine& l){ return l.dying; }), lines.end());
}
