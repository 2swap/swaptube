#include "AffineScene.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "../../IO/Latex.h"

using std::ostringstream;
using std::ostringstream;
using std::fixed;
using std::setprecision;

string double_to_string_2(double value) {
    const int sig_figs = 2;
    ostringstream out;
    if (value == 0) { return "0"; }

    int exponent = static_cast<int>(std::floor(std::log10(std::abs(value))));
    int significant_digits = max(0, sig_figs - exponent - 1);

    out << std::fixed << std::setprecision(significant_digits) << value;
    string str = out.str();

    if (significant_digits > 0) {
        // Remove trailing zeros
        str.erase(str.find_last_not_of('0') + 1, string::npos);

        // Remove decimal point if it is the last character
        if (!str.empty() && str.back() == '.') {
            str.pop_back();
        }
    }

    return str;
}

extern "C" void cuda_draw_point(uint32_t* d_pixels, ivec2 wh, vec2 pos, float opacity, float thickness, uint32_t color, vec4 matrix, int style);
extern "C" void cuda_draw_line(uint32_t* d_pixels, ivec2 wh, vec2 pos1, vec2 pos2, float opacity, float thickness, uint32_t color1, uint32_t color2, int style);
//extern "C" void cuda_draw_icon(uint32_t* d_pixels, ivec2 wh, vec2 pos, float opacity, float thickness, uint32_t color);
extern "C" void cuda_overlay_linear (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const vec4& inv_matrix);

AffineScene::AffineScene(uint32_t* icons, ivec4& icons_whnm, int& icons_len, const vec2& dimension)
: Scene(dimension), icons(icons), icons_whnm(icons_whnm), icons_len(icons_len) {
    manager.set({
        {"r", "1"},
        {"g", "1"},
        {"b", "1"},
    });
}

void AffineScene::reset() {
    num_points = 0;
    line_endpoints = {};
    text_anchors = {};
    text_latex = {};
    icon_anchors = {};
    draw_order = {};
}

void AffineScene::reorder_layers(vector<AffineElementType> types) {
    draw_order = {};
    for (int t=0; t<types.size(); t++) {
        if (types[t] == POINT) for (int i=0; i<num_points; i++) {
            draw_order.push_back({POINT, i});
        } else if (types[t] == LINE) for (int i=0; i<line_endpoints.size(); i++) {
            draw_order.push_back({LINE, i});
        } else if (types[t] == TEXT) for (int i=0; i<text_anchors.size(); i++) {
            draw_order.push_back({TEXT, i});
        } else for (int i=0; i<icon_anchors.size(); i++) {
            draw_order.push_back({ICON, i});
        }
    }
}

int AffineScene::add_point(vector<int> anchors) {
    int id = num_points;
    num_points++;
    draw_order.push_back({POINT, id});
    string s = std::to_string(id);
    string xeq = "<p"+s+"x> 0 ";
    string yeq = "<p"+s+"y> 0 ";
    string weq = "0 ";
    if (anchors.size() > 0) {
        for (int i=0; i<anchors.size(); i++) {
            manager.set("p"+s+"w"+std::to_string(anchors[i]), "1");
            xeq += "<p" + std::to_string(anchors[i]) + "xeq> <p"+s+"w" + std::to_string(anchors[i]) + "> * + ";
            yeq += "<p" + std::to_string(anchors[i]) + "yeq> <p"+s+"w" + std::to_string(anchors[i]) + "> * + ";
            weq += "<p"+s+"w" + std::to_string(anchors[i]) + "> + ";
        }
        xeq += weq + "/ ";
        yeq += weq + "/ ";
    }
    manager.set({
        {"p"+s+"xeq", xeq + "+"},
        {"p"+s+"yeq", yeq + "+"},
        {"p"+s+"x", "0"},
        {"p"+s+"y", "0"},
        {"p"+s+"o", "1"},
        {"p"+s+"t", "0.01"},
        {"p"+s+"r", "<r>"},
        {"p"+s+"g", "<g>"},
        {"p"+s+"b", "<b>"},
        {"p"+s+"m00", "1"},
        {"p"+s+"m10", "0"},
        {"p"+s+"m01", "0"},
        {"p"+s+"m11", "1"},
        {"p"+s+"s", "0"},
    });
    return id;
}

vec2 AffineScene::get_pos(int point) {
    return vec2(state["p" + std::to_string(point) + "xeq"], state["p" + std::to_string(point) + "yeq"]);
}

int AffineScene::add_line(int point1, int point2) {
    int id = line_endpoints.size();
    draw_order.push_back({LINE, id});
    line_endpoints.push_back(ivec2(point1, point2));
    manager.set({
        {"l" + std::to_string(id) + "x", "0"},
        {"l" + std::to_string(id) + "y", "0"},
        {"l" + std::to_string(id) + "o", "1"},
        {"l" + std::to_string(id) + "s", "0"},
        {"l" + std::to_string(id) + "t", "0.005"},
        {"l" + std::to_string(id) + "r1", "<r>"},
        {"l" + std::to_string(id) + "g1", "<g>"},
        {"l" + std::to_string(id) + "b1", "<b>"},
        {"l" + std::to_string(id) + "r2", "<r>"},
        {"l" + std::to_string(id) + "g2", "<g>"},
        {"l" + std::to_string(id) + "b2", "<b>"},
        {"l" + std::to_string(id) + "s", "0"},
    });
    return id;
}

int AffineScene::add_text(int anchor, vector<string> latex_bits, bool hide_point) {
    int id = text_anchors.size();
    string s = std::to_string(id);
    draw_order.push_back({TEXT, id});
    text_anchors.push_back(anchor);
    text_latex.push_back(latex_bits);
    manager.set({
        {"t"+s+"x", "0"},
        {"t"+s+"y", "0"},
        {"t"+s+"o", "1"},
        //{"t"+s+"a", "0"},
        {"t"+s+"m00", "1"},
        {"t"+s+"m10", "0"},
        {"t"+s+"m01", "0"},
        {"t"+s+"m11", "1"},
        {"t"+s+"h", "0.1"},

        {"p" + std::to_string(anchor) + "o", std::to_string(1 - hide_point)},
    });
    for (int i=0; i<latex_bits.size()-1; i++) {
        manager.set("t"+s+"v" + std::to_string(i), "0");
    }
    return id;
}

int AffineScene::add_icon(int anchor) {
    int id = icon_anchors.size();
    draw_order.push_back({ICON, id});
    icon_anchors.push_back(anchor);
    manager.set({
        {"i" + std::to_string(id) + "x", "0"},
        {"i" + std::to_string(id) + "y", "0"},
        {"i" + std::to_string(id) + "o", "1"},
        {"i" + std::to_string(id) + "h", "0.1"},
        {"i" + std::to_string(id) + "m00", "1"},
        {"i" + std::to_string(id) + "m10", "0"},
        {"i" + std::to_string(id) + "m01", "0"},
        {"i" + std::to_string(id) + "m11", "1"},
        {"i" + std::to_string(id) + "id", "1"},

        {"p" + std::to_string(anchor) + "o", "0"},
    });
    return id;
}

void AffineScene::draw_point(int id) {
    if (state["p" + std::to_string(id) + "o"] < 0.01) return;
    ivec2 wh = get_width_height();
    cuda_draw_point(
        gpu_pix.get_ptr(), wh,
        get_pos(id), clamp(state["p" + std::to_string(id) + "o"], 0, 1), state["p" + std::to_string(id) + "t"] * wh.y,
        0xff000000
        | (uint32_t(fminf(255, 256 * state["p" + std::to_string(id) + "r"])) << 16)
        | (uint32_t(fminf(255, 256 * state["p" + std::to_string(id) + "g"])) << 8)
        |  uint32_t(fminf(255, 256 * state["p" + std::to_string(id) + "b"])),
        vec4(state["p" + std::to_string(id) + "m00"], state["p" + std::to_string(id) + "m10"], state["p" + std::to_string(id) + "m01"], state["p" + std::to_string(id) + "m11"]),
        state["p" + std::to_string(id) + "s"]
    );
}

void AffineScene::draw_line(int id) {
    if (state["l" + std::to_string(id) + "o"] < 0.01) return;
    ivec2 wh = get_width_height();
    vec2 offset(state["l" + std::to_string(id) + "x"], state["l" + std::to_string(id) + "y"]);
    cuda_draw_line(
        gpu_pix.get_ptr(), wh,
        get_pos(line_endpoints[id].x) + offset, get_pos(line_endpoints[id].y) + offset, clamp(state["l" + std::to_string(id) + "o"], 0, 1), state["l" + std::to_string(id) + "t"] * wh.y,
        0xff000000
        | (uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "r1"])) << 16)
        | (uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "g1"])) << 8)
        |  uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "b1"])),
        0xff000000
        | (uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "r2"])) << 16)
        | (uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "g2"])) << 8)
        |  uint32_t(fminf(255, 256 * state["l" + std::to_string(id) + "b2"])),
        state["l" + std::to_string(id) + "s"]
    );
}

void AffineScene::draw_text(int id) {
    if (state["t" + std::to_string(id) + "o"] < 0.01) return;
    const ivec2 wh = get_width_height();
    ScalingParams sp(wh * vec2(1, .6));
    string latex_full = text_latex[id][0];
    for (int i=0; i<text_latex[id].size()-1; i++) {
        latex_full = latex_full + double_to_string_2(state["t" + std::to_string(id) + "v" + std::to_string(i)]) + text_latex[id][i + 1];
    }
    write_text_linear(
        gpu_pix.get_ptr(), wh, latex_full,
        (get_pos(text_anchors[id]) + vec2(state["t" + std::to_string(id) + "x"], state["t" + std::to_string(id) + "y"])) * wh,
        vec2(1, state["t" + std::to_string(id) + "h"]) * wh, clamp(state["t" + std::to_string(id) + "o"], 0, 1),
        vec4(state["t" + std::to_string(id) + "m00"], state["t" + std::to_string(id) + "m10"], state["t" + std::to_string(id) + "m01"], state["t" + std::to_string(id) + "m11"])
    );
}

void AffineScene::draw_icon(int id) {
    if (state["i" + std::to_string(id) + "o"] < 0.01 || state["i" + std::to_string(id) + "id"] < 0 || state["i" + std::to_string(id) + "id"] >= icons_len) return;
    ivec2 wh = get_width_height();
    float scale = wh.y * state["i" + std::to_string(id) + "h"] / icons_whnm.y;
    cuda_overlay_linear(
        gpu_pix.get_ptr(), wh,
        icons+(uint32_t)(state["i" + std::to_string(id) + "id"])*icons_whnm.x*icons_whnm.y, ivec2(icons_whnm.x, icons_whnm.y),
        (get_pos(icon_anchors[id]) + vec2(state["i" + std::to_string(id) + "x"], state["i" + std::to_string(id) + "y"])) * wh, clamp(state["i" + std::to_string(id) + "o"], 0, 1),
        vec4(state["i" + std::to_string(id) + "m00"], state["i" + std::to_string(id) + "m10"], state["i" + std::to_string(id) + "m01"], state["i" + std::to_string(id) + "m11"]) / scale
    );
    /*cuda_draw_point(
        gpu_pix.get_ptr(), wh,
        get_pos(icon_anchors[id]) + vec2(state["i" + std::to_string(id) + "x"], state["i" + std::to_string(id) + "y"]), clamp(state["i" + std::to_string(id) + "o"], 0, 1), state["i" + std::to_string(id) + "h"] * wh.y / 2,
        0xff000000
        | (uint32_t(fminf(255, 256 * state["r"])) << 16)
        | (uint32_t(fminf(255, 256 * state["g"])) << 8)
        |  uint32_t(fminf(255, 256 * state["b"])),
        vec4(1,0,0,1), 2
    );*/
}

void AffineScene::draw() {
    for (int i=0; i<draw_order.size(); i++) {
        if (draw_order[i].elemtype == POINT) {
            draw_point(draw_order[i].index);
        } else if (draw_order[i].elemtype == LINE) {
            draw_line(draw_order[i].index);
        } else if (draw_order[i].elemtype == TEXT) {
            draw_text(draw_order[i].index);
        } else {
            draw_icon(draw_order[i].index);
        }
    }
}
