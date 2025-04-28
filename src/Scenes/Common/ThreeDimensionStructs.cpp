// TODO includes for GLM
enum NodeHighlightType {
    NORMAL,
    RING,
    BULLSEYE,
};

struct Point {
    glm::vec3 center;
    int color;
    float opacity;
    NodeHighlightType highlight;
    float size;
    Point(const glm::vec3& pos, int clr, NodeHighlightType hlt = NORMAL, float op = 1, float sz = 1)
        : highlight(hlt), size(sz) { center = pos; color = clr; opacity = op; }
};

struct Line {
    int color;
    float opacity;
    glm::vec3 start;
    glm::vec3 end;
    Line(const glm::vec3& s, const glm::vec3& e, int clr, float op = 1)
        : color(clr), opacity(op), start(s), end(e) {}
};

