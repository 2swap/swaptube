#include "Color.h"
#include "vec.h"
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

struct Point {
    vec3 center;
    Color color;
    float opacity;
    float size;
    Point(const vec3& pos, Color clr, float op = 1, float sz = 1)
        : center(pos), color(clr), opacity(op), size(sz) { }
};

struct Line {
    Color color;
    float opacity;
    vec3 start;
    vec3 end;
    Line(const vec3& s, const vec3& e, Color clr, float op = 1)
        : color(clr), opacity(op), start(s), end(e) { }
};

SHARED_FILE_SUFFIX
