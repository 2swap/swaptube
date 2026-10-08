#pragma once

#include "../Scene.h"
#include <string>
#include <vector>

using std::string;
using std::vector;

enum AffineElementType {
    POINT,
    LINE,
    TEXT,
    ICON
};

struct AffineElement {
    AffineElementType elemtype;
    int index;
};

class AffineScene : public Scene {
public:
    AffineScene(uint32_t* icons, ivec4& icons_whnm, int& icons_len, const vec2& dimension = vec2(1, 1));

    vector<AffineElement> draw_order;

    int add_point(vector<int> anchors = {});
    int add_line(int point1, int point2);
    int add_text(int anchor, vector<string> latex_bits, bool hide_point = true);
    int add_icon(int anchor);
    vec2 get_pos(int point);
    void reorder_layers(vector<AffineElementType> types);
    void reset();

    int num_points = 0;
    vector<ivec2> line_endpoints;
    vector<int> text_anchors;
    vector<vector<string>> text_latex;
    vector<int> icon_anchors;

private:
    uint32_t* icons;
    ivec4& icons_whnm;
    int& icons_len;

    void draw_point(int id);
    void draw_line(int id);
    void draw_text(int id);
    void draw_icon(int id);
    void draw() override;
};

