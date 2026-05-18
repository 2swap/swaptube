#include "vec.h"
#include "shared_precompiler_directives.h"

SHARED_FILE_PREFIX

struct Point {
    vec3 center;
    int color;
    float opacity;
    float size;
    Point(const vec3& pos, int clr, float op = 1, float sz = 1)
        : center(pos), color(clr), opacity(op), size(sz) { }
};

struct Line {
    int color;
    float opacity;
    vec3 start;
    vec3 end;
    bool is_dashed;
    Line(const vec3& s, const vec3& e, int clr, float op, bool dashed)
        : color(clr), opacity(op), start(s), end(e), is_dashed(dashed) { }
};

SHARED_FILE_SUFFIX
