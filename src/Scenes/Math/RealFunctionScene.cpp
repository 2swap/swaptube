#pragma once

#include "../Common/CoordinateSceneWithTrail.cpp"
#include <vector>
#include <stdexcept>
#include <string>

struct FunctionData {
    std::string function;
    std::string transition_function;
    uint32_t color;

    FunctionData(const std::string& f, const std::string& tf, uint32_t c)
        : function(f), transition_function(tf), color(c) {}
};

class RealFunctionScene : public CoordinateSceneWithTrail {
public:
    RealFunctionScene(const double width = 1, const double height = 1)
        : CoordinateSceneWithTrail(width, height), pixel_width(0.1) {}

    // Evaluates the function at x by replacing '?' with the x-value,
    // computing the current function and its transitional variant, then smoothing between them.
    double call_the_function(double x, const FunctionData& func_data) {
        double stf = state["microblock_fraction"];
        string xstring = to_string(x);
        string f1 = replace_substring(func_data.function, "?", xstring);
        string f2 = replace_substring(func_data.transition_function, "?", xstring);
        double y1 = calculator(f1);
        double y2 = !f2.empty() ? calculator(f2) : y1;
        return smoothlerp(y1, y2, stf);
    }

    void begin_transition(int index, const string& s) {
        if (index < 0 || index >= functions.size())
            throw runtime_error("Bad function index!");
        functions[index].transition_function = s;
    }

    void draw() override {
        CoordinateSceneWithTrail::draw();
        render_functions();
    }

    void add_function(const string& func, uint32_t color) {
        functions.emplace_back(func, "", color);
    }

    void set_pixel_width(double w) {
        pixel_width = w;
    }

    double get_pixel_width() const {
        return pixel_width;
    }

    void render_functions() {
        double left_bound  = state["left_x"];
        double right_bound = state["right_x"];
        double dx = (right_bound - left_bound) / static_cast<double>(get_width());
        double last_subtr = 0;
        for (double x = left_bound; x <= right_bound; x += dx) {
            double subtr = 0;
            int mult = 1;
            double val = 0;
            for (const auto& func_data : functions) {
                val = call_the_function(x, func_data);
                pair<int, int> pixel = point_to_pixel({x, val});
                render_dot(pixel, func_data.color);
                subtr += val * mult;
                mult = -1;
            }
            if (last_subtr * subtr < 0 && functions.size() == 2) {
                pair<int, int> collide = point_to_pixel({x, val});
                render_point(collide);
                pix.bresenham(collide.first, collide.second, collide.first, get_height() / 2, 0xffffffff, 1, 2);
            }
            last_subtr = subtr;
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateSceneWithTrail::populate_state_query();
        sq.insert("microblock_fraction");
        return sq;
    }

    void mark_data_unchanged() override {}
    void change_data() override {} // RealFunctionScene has no DataObjects
    bool check_if_data_changed() const override { return false; }

    // On ending a transition, update each function if it was transitioning.
    void on_end_transition_extra_behavior(bool is_macroblock) override {
        for (int i = 0; i < functions.size(); i++) {
            if (!functions[i].transition_function.empty())
                functions[i].function = functions[i].transition_function;
            functions[i].transition_function = "";
        }
    }

private:
    double pixel_width;
    vector<FunctionData> functions;
};

