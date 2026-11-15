#include "Latex.h"

#include <vector>
#include <array>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <librsvg-2.0/librsvg/rsvg.h>
#include <sys/stat.h>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <limits.h>
#include <unistd.h>
#include <cairo.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include "../Core/Pixels.h"

using namespace std;

extern "C" void cuda_overlay (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const float angle_rad);

extern "C" void cuda_overlay_linear (
    uint32_t* background, const ivec2& b_wh,
    const uint32_t* foreground, const ivec2& f_wh,
    const vec2& center, const float opacity, const vec4& inv_matrix);

string latex_color(uint32_t color, string text) {
    // Mask out the alpha channel
    uint32_t rgb = color & 0x00FFFFFF;

    // Convert to a hex string
    stringstream ss;
    ss << "\\textcolor{#" << hex << setw(6) << setfill('0') << rgb << "}{" << text << "}";

    return ss.str();
}

// Custom hash and equality for pair<string, pair<int,int>>
struct StringIntPairHash {
    size_t operator()(const pair<string, pair<int, int>>& p) const noexcept {
        size_t h1 = std::hash<string>{}(p.first);
        size_t h2 = (static_cast<size_t>(p.second.first) << 32) ^ static_cast<size_t>(p.second.second);
        // boost-like mix
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
    }
};
struct StringIntPairEq {
    bool operator()(const pair<string, pair<int, int>>& a, const pair<string, pair<int, int>>& b) const noexcept {
        return a.first == b.first && a.second.first == b.second.first && a.second.second == b.second.second;
    }
};

// Create an unordered_map to store the cached results
unordered_map<string, shared_ptr<DevicePointer>> latex_cache;

static string generate_cache_key(const string& text, const ScalingParams& scaling_params) {
    hash<string> hasher;
    string key = text + "_" + to_string(static_cast<int>(scaling_params.mode)) + "_" + 
                 to_string(scaling_params.bounding_box.x) + "_" + 
                 to_string(scaling_params.bounding_box.y) + "_" +
                 to_string(scaling_params.scale_factor);
    return to_string(hasher(key));
}

// Makes sure MicroTeX has rendered this equation to io_in/latex/, and returns
// the file's path relative to io_in/ ("latex/<hash>.svg").
static string ensure_latex_svg(const string& latex) {
    hash<string> hasher;
    char full_directory_path[PATH_MAX];
    string latex_dir = "io_in/latex/";
    realpath(latex_dir.c_str(), full_directory_path);
    string name_without_folder = to_string(hasher(latex)) + ".svg";
    string name = string(full_directory_path) + "/" + name_without_folder;

    if (access(name.c_str(), F_OK) == -1) {
        string command = "cd ../../MicroTeX-master/build/ && ./LaTeX -headless -foreground=#ffffffff \"-input=" + latex + "\" -output=" + name + " >/dev/null 2>&1";
        int result = system(command.c_str());
        if(result != 0) {
            cout << command << endl;
            throw runtime_error("Failed to generate LaTeX. Command printed above.");
        }
    }
    return "latex/" + name_without_folder;
}

/*
 * We use MicroTEX to convert LaTeX equations into svg files.
 */
shared_ptr<DevicePointer> latex_to_gpu_pix(const string& latex, ScalingParams& scaling_params) {
    // Generate a cache key based on the equation and scaling parameters
    string cache_key = generate_cache_key(latex, scaling_params);

    // Check if the result is already in the cache
    auto it = latex_cache.find(cache_key);
    if (it != latex_cache.end()) {
        return it->second; // Return the cached result
    }

    // cout << "Generating LaTeX for: " << latex << endl;

    const string svg_path = ensure_latex_svg(latex);

    // System call successful, return the generated SVG
    shared_ptr<DevicePointer> text = svg_to_gpu_pix(svg_path, scaling_params);
                // cout << "Dimensions = " << text->get_wh().x << " x " << text->get_wh().y << endl;
    latex_cache[cache_key] = text; // Cache the result before returning
    return text;
}

// ------------------
// Glyph ID rendering
// ------------------
// MicroTeX's SVGs (written by Cairo) draw every font glyph as its own <use> element, and every
// rule (fraction bar, \hline, table border segment, radical overline, \overline...) as its own
// stroked <path>. We rewrite the SVG so each drawn element gets a unique flat color encoding its
// index, and turn off antialiasing, so that rasterizing it yields a per-pixel element ID map.
// What each element is (which glyph, or which kind of rule) is read separately from the SVG: see latex_glyph_shapes.

// Largest element ID that fits in a pixel's RGB.
static const uint32_t MAX_ELEMENT_ID = 0xFFFFFF;

// Rules are stroked, unfilled paths. Cairo writes this as a style property, or in older versions as an attribute.
static bool is_rule_tag(const string& tag) {
    return tag.find("fill:none") != string::npos || tag.find("fill=\"none\"") != string::npos;
}

static string id_color_hex(uint32_t id) {
    stringstream ss;
    ss << "#" << hex << setw(6) << setfill('0') << (id & 0xFFFFFF);
    return ss.str();
}

// If element_tags is given, it is filled so that (*element_tags)[id] is the original SVG tag of element `id`
// (a <use> for font glyphs, a <path> etc. for rules and other drawables). Index 0 is unused.
static string make_glyph_id_svg(const string& svg, vector<string>* element_tags = nullptr) {
    const size_t defs_end = svg.find("</defs>");
    size_t body_start = (defs_end == string::npos) ? svg.find('>', svg.find("<svg")) + 1 : defs_end + 7;
    string head = svg.substr(0, body_start);
    const string body = svg.substr(body_start);

    // Disable antialiasing so every covered pixel carries an exact ID color.
    const size_t svg_tag = head.find("<svg");
    if (svg_tag != string::npos) head.insert(svg_tag + 4, " shape-rendering=\"crispEdges\"");

    static const regex color_prop(R"((fill|stroke):(rgb\([^)]*\)|#[0-9a-fA-F]+))");
    static const regex color_attr(R"(\b(fill|stroke)="(rgb\([^)]*\)|#[0-9a-fA-F]+)\")");
    static const regex style_attr(R"(style="([^"]*)\")");
    static const regex drawable(R"(^<(path|rect|circle|ellipse|polygon|polyline|line)\b)");

    if (element_tags) element_tags->assign(1, "");
    string out;
    out.reserve(body.size() + 4096);
    uint32_t next_id = 1;
    size_t pos = 0;
    while (pos < body.size()) {
        const size_t lt = body.find('<', pos);
        if (lt == string::npos) { out += body.substr(pos); break; }
        out += body.substr(pos, lt - pos);
        const size_t gt = body.find('>', lt);
        if (gt == string::npos) { out += body.substr(lt); break; }
        string tag = body.substr(lt, gt - lt + 1);

        const bool is_use = tag.rfind("<use", 0) == 0;
        if (is_use || regex_search(tag, drawable)) {
            if (next_id > MAX_ELEMENT_ID) throw runtime_error("make_glyph_id_svg: too many elements");
            if (element_tags) element_tags->push_back(tag);
            const string color = id_color_hex(next_id);
            if (is_use) {
                // Font glyph: the referenced <symbol> path has no fill of its own and inherits this one.
                tag.insert(4, " style=\"fill:" + color + ";stroke:none;\"");
            } else {
                const bool is_rule = is_rule_tag(tag);
                smatch m;
                string style = regex_search(tag, m, style_attr) ? m[1].str() : "";
                string new_style = regex_replace(style, color_prop, "$1:" + color);
                if (!is_rule && new_style.find("fill:") == string::npos) new_style = "fill:" + color + ";" + new_style;
                if (m.empty()) tag.insert(tag.find(' ') == string::npos ? tag.size() - 1 : tag.find(' '), " style=\"" + new_style + "\"");
                else tag.replace(m.position(1), m.length(1), new_style);
                tag = regex_replace(tag, color_attr, "$1=\"" + color + "\"");   // Colors given as attributes
            }
            next_id++;
        }
        out += tag;
        pos = gt + 1;
    }
    return head + out;
}

unordered_map<string, Pixels> glyph_id_cache;

Pixels latex_to_glyph_ids(const string& latex, const ScalingParams& scaling_params) {
    const string cache_key = generate_cache_key(latex, scaling_params);
    auto it = glyph_id_cache.find(cache_key);
    if (it != glyph_id_cache.end()) return it->second;

    const string svg_path = ensure_latex_svg(latex);
    const string id_svg_path = svg_path.substr(0, svg_path.size() - 4) + "_element_ids.svg";
    const string id_svg_full = "io_in/" + id_svg_path;
    if (access(id_svg_full.c_str(), F_OK) == -1) {
        ifstream in("io_in/" + svg_path);
        if (!in) throw runtime_error("latex_to_glyph_ids: could not read io_in/" + svg_path);
        stringstream buffer;
        buffer << in.rdbuf();
        ofstream out(id_svg_full);
        if (!out) throw runtime_error("latex_to_glyph_ids: could not write " + id_svg_full);
        out << make_glyph_id_svg(buffer.str());
    }

    Pixels ids = svg_to_aligned_pixels(svg_path, id_svg_path, scaling_params);
    glyph_id_cache[cache_key] = ids;
    return ids;
}

// ------------------
// Glyph shape identity
// ------------------
// Every <use> element draws a <symbol> whose path is a font glyph's outline, anchored at the glyph origin.
// The same character at another size is the same outline scaled about that origin (a superscript is x0.7, a
// second-level superscript x0.5), so outlines identify symbols across sizes and across equations.
// The path data can't be compared directly: Cairo rounds coordinates to a fine grid, so very short segments
// (real or not) vanish at some sizes and not others, changing the segment structure. So we compare the drawn
// shapes instead: flatten each outline to a polyline, normalize it by its bounding box diagonal, and call two
// outlines the same if every point of each lies within SHAPE_TOLERANCE of the other (Hausdorff distance).
// Measured on real equations: copies of one glyph at different sizes differ by at most ~0.002, while the
// closest distinct glyphs (e.g. italic vs upright C) differ by ~0.07.

// Max Hausdorff distance between two normalized outlines of the same glyph, as a fraction of the bounding box diagonal.
static const double SHAPE_TOLERANCE = 0.01;
// Line segments each cubic curve is flattened into.
static const int CURVE_SAMPLES = 8;

struct GlyphOutline {
    int num_contours = 0;
    vector<array<double, 4>> segments;  // Flattened outline, normalized: x0, y0, x1, y1
    double min_x = 0, min_y = 0, max_x = 0, max_y = 0; // Normalized bounding box
};

// Parses Cairo's absolute M/L/C/Z path data into a normalized polyline outline.
// Returns false for other path commands, or an empty outline.
static bool parse_glyph_outline(const string& d, GlyphOutline& out) {
    out = GlyphOutline();
    size_t i = 0;
    auto skip_separators = [&]() { while (i < d.size() && (isspace((unsigned char) d[i]) || d[i] == ',')) i++; };
    auto read_number = [&](double& v) {
        skip_separators();
        if (i >= d.size()) return false;
        const char* start = d.c_str() + i;
        char* end;
        v = strtod(start, &end);
        if (end == start) return false;
        i += end - start;
        return true;
    };
    double cx = 0, cy = 0, sx = 0, sy = 0;   // Current point and contour start
    char command = 0;
    while (true) {
        skip_separators();
        if (i >= d.size()) break;
        if (isalpha((unsigned char) d[i])) command = d[i++];
        else if (command == 0) return false;   // Coordinates before any command
        // (Otherwise a repeated command with its letter omitted, which SVG allows.)

        if (command == 'Z') {
            out.segments.push_back({cx, cy, sx, sy});
            cx = sx; cy = sy;
            command = 0;   // Z takes no coordinates, so it can't repeat implicitly
        } else if (command == 'M') {
            if (!read_number(cx) || !read_number(cy)) return false;
            sx = cx; sy = cy;
            out.num_contours++;
            command = 'L';   // Coordinates after an M are implicit L's
        } else if (command == 'L') {
            double x, y;
            if (!read_number(x) || !read_number(y)) return false;
            out.segments.push_back({cx, cy, x, y});
            cx = x; cy = y;
        } else if (command == 'C') {
            double p[6];
            for (double& v : p) if (!read_number(v)) return false;
            double px = cx, py = cy;
            for (int k = 1; k <= CURVE_SAMPLES; k++) {
                const double u = (double) k / CURVE_SAMPLES, w = 1 - u;
                const double x = w*w*w*cx + 3*w*w*u*p[0] + 3*w*u*u*p[2] + u*u*u*p[4];
                const double y = w*w*w*cy + 3*w*w*u*p[1] + 3*w*u*u*p[3] + u*u*u*p[5];
                out.segments.push_back({px, py, x, y});
                px = x; py = y;
            }
            cx = p[4]; cy = p[5];
        } else {
            return false;   // Relative commands, arcs etc.: Cairo doesn't emit them
        }
    }
    if (out.segments.empty()) return false;

    // Normalize by the bounding box diagonal. Not translated: outlines are anchored at the glyph origin,
    // and scaling about that origin is exactly how sizes differ.
    double min_x = out.segments[0][0], max_x = min_x, min_y = out.segments[0][1], max_y = min_y;
    for (const auto& s : out.segments) for (int k = 0; k < 4; k += 2) {
        min_x = min(min_x, s[k]); max_x = max(max_x, s[k]);
        min_y = min(min_y, s[k + 1]); max_y = max(max_y, s[k + 1]);
    }
    const double diag = hypot(max_x - min_x, max_y - min_y);
    if (diag == 0) return false;
    for (auto& s : out.segments) for (double& v : s) v /= diag;
    out.min_x = min_x / diag; out.max_x = max_x / diag;
    out.min_y = min_y / diag; out.max_y = max_y / diag;
    return true;
}

// Largest distance from a point of a's polyline to the nearest point of b's polyline.
static double directed_hausdorff(const GlyphOutline& a, const GlyphOutline& b) {
    double worst = 0;
    for (const auto& sa : a.segments) {
        const double px = sa[0], py = sa[1];   // Every vertex of a starts some segment
        double best = numeric_limits<double>::infinity();
        for (const auto& sb : b.segments) {
            const double dx = sb[2] - sb[0], dy = sb[3] - sb[1];
            const double len2 = dx * dx + dy * dy;
            const double t = len2 == 0 ? 0 : max(0.0, min(1.0, ((px - sb[0]) * dx + (py - sb[1]) * dy) / len2));
            best = min(best, hypot(px - (sb[0] + t * dx), py - (sb[1] + t * dy)));
            if (best <= worst) break;   // Can't raise the maximum
        }
        worst = max(worst, best);
        if (worst > SHAPE_TOLERANCE) return worst;   // Already a mismatch
    }
    return worst;
}

static bool same_shape(const GlyphOutline& a, const GlyphOutline& b) {
    if (a.num_contours != b.num_contours) return false;
    // Cheap rejection: normalized bounding boxes of the same shape coincide
    if (abs(a.min_x - b.min_x) > SHAPE_TOLERANCE || abs(a.max_x - b.max_x) > SHAPE_TOLERANCE ||
        abs(a.min_y - b.min_y) > SHAPE_TOLERANCE || abs(a.max_y - b.max_y) > SHAPE_TOLERANCE) return false;
    return directed_hausdorff(a, b) <= SHAPE_TOLERANCE && directed_hausdorff(b, a) <= SHAPE_TOLERANCE;
}

// Rules have no outline to compare, so they're classed by orientation: a fraction bar can become another
// fraction bar of any length, but not a vertical table border.
static const uint32_t HORIZONTAL_RULE_SHAPE = 1;
static const uint32_t VERTICAL_RULE_SHAPE = 2;
static const uint32_t FIRST_GLYPH_SHAPE = 3;

// Orientation class of a rule tag, from its path's extent. (Its transform is a uniform scale, which doesn't
// change orientation.) 0 if it has no path data.
static uint32_t rule_shape(const string& tag) {
    static const regex d_attr(R"(\bd="([^"]*)\")");
    smatch m;
    if (!regex_search(tag, m, d_attr)) return 0;
    const string d = m[1].str();
    static const regex number(R"(-?\d*\.?\d+(?:[eE][-+]?\d+)?)");
    vector<double> v;
    for (sregex_iterator it(d.begin(), d.end(), number), end; it != end; ++it) v.push_back(stod((*it)[0].str()));
    if (v.size() < 2) return 0;
    double min_x = v[0], max_x = v[0], min_y = v[1], max_y = v[1];
    for (size_t k = 0; k + 1 < v.size(); k += 2) {
        min_x = min(min_x, v[k]); max_x = max(max_x, v[k]);
        min_y = min(min_y, v[k + 1]); max_y = max(max_y, v[k + 1]);
    }
    return (max_x - min_x) >= (max_y - min_y) ? HORIZONTAL_RULE_SHAPE : VERTICAL_RULE_SHAPE;
}

// Shape classes for the whole run, so that class numbers agree across equations.
// Each class keeps the first outline seen as its representative. Identical path data (the same glyph at the
// same size, the common case) is looked up directly, skipping the geometric comparison.
static uint32_t shape_class(const string& path_data) {
    static unordered_map<string, uint32_t> by_path_data;
    static vector<pair<GlyphOutline, uint32_t>> representatives;
    static uint32_t next_class = FIRST_GLYPH_SHAPE;

    auto known = by_path_data.find(path_data);
    if (known != by_path_data.end()) return known->second;

    uint32_t id = 0;   // 0 if the outline can't be read
    GlyphOutline outline;
    if (parse_glyph_outline(path_data, outline)) {
        for (const auto& [representative, rep_id] : representatives)
            if (same_shape(representative, outline)) { id = rep_id; break; }
        if (id == 0) {
            id = next_class++;
            representatives.emplace_back(move(outline), id);
        }
    }
    by_path_data[path_data] = id;
    return id;
}

// Maps each <symbol> id in the SVG's <defs> to the path data it contains (all its paths' d attributes joined).
static unordered_map<string, string> read_symbol_paths(const string& svg) {
    unordered_map<string, string> symbols;
    static const regex id_attr(R"(\bid="([^"]+)\")");
    static const regex d_attr(R"(\bd="([^"]*)\")");
    size_t pos = 0;
    while ((pos = svg.find("<symbol", pos)) != string::npos) {
        const size_t open_end = svg.find('>', pos);
        const size_t close = svg.find("</symbol>", pos);
        if (open_end == string::npos || close == string::npos) break;
        const string open_tag = svg.substr(pos, open_end - pos + 1);
        smatch m;
        if (regex_search(open_tag, m, id_attr)) {
            const string id = m[1].str();
            string paths;
            const string inner = svg.substr(open_end + 1, close - open_end - 1);
            for (sregex_iterator it(inner.begin(), inner.end(), d_attr), end; it != end; ++it)
                paths += (*it)[1].str() + " ";
            symbols[id] = paths;
        }
        pos = close + 9;
    }
    return symbols;
}

unordered_map<string, vector<uint32_t>> glyph_shape_cache;

vector<uint32_t> latex_glyph_shapes(const string& latex) {
    auto it = glyph_shape_cache.find(latex);
    if (it != glyph_shape_cache.end()) return it->second;

    const string svg_path = ensure_latex_svg(latex);
    ifstream in("io_in/" + svg_path);
    if (!in) throw runtime_error("latex_glyph_shapes: could not read io_in/" + svg_path);
    stringstream buffer;
    buffer << in.rdbuf();
    const string svg = buffer.str();

    // Same walk that numbers the elements of the glyph ID image, so indices line up with its IDs
    vector<string> element_tags;
    make_glyph_id_svg(svg, &element_tags);
    const unordered_map<string, string> symbols = read_symbol_paths(svg);

    static const regex href_attr(R"(href="#([^"]+)\")");
    vector<uint32_t> shapes(element_tags.size(), 0);
    for (size_t id = 1; id < element_tags.size(); id++) {
        const string& tag = element_tags[id];
        smatch m;
        if (tag.rfind("<use", 0) == 0) {
            if (!regex_search(tag, m, href_attr)) continue;
            auto sym = symbols.find(m[1].str());
            if (sym != symbols.end()) shapes[id] = shape_class(sym->second);
        } else if (is_rule_tag(tag)) {
            shapes[id] = rule_shape(tag);
        }
    }
    glyph_shape_cache[latex] = shapes;
    return shapes;
}

void write_text(uint32_t* gpu_pix, const ivec2& canvas_wh, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const float angle_rad) {
    ScalingParams scaling_params(text_envelope);

    shared_ptr<DevicePointer> text = latex_to_gpu_pix(latex, scaling_params);
    cuda_overlay(gpu_pix, canvas_wh, text->get_ptr(), text->get_wh(), center, opacity, angle_rad);
}

void write_text_linear(uint32_t* gpu_pix, const ivec2& canvas_wh, const std::string& latex, const vec2& center, const vec2& text_envelope, const double opacity, const vec4& inv_matrix) {
    ScalingParams scaling_params(text_envelope);

    shared_ptr<DevicePointer> text = latex_to_gpu_pix(latex, scaling_params);
    cuda_overlay_linear(gpu_pix, canvas_wh, text->get_ptr(), text->get_wh(), center, opacity, inv_matrix);
}
