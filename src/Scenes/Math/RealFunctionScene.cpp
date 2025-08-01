#pragma once

#include "../Common/CoordinateScene.cpp"
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

class RealFunctionScene : public CoordinateScene {
public:
    RealFunctionScene(const double width = 1, const double height = 1)
        : CoordinateScene(width, height), pixel_width(0.1) {}

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
        CoordinateScene::draw();
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
                glm::vec2 pixel = point_to_pixel({x, val});
                float sz = get_geom_mean_size() / 200.f;
                pix.fill_rect(pixel.x - sz / 2, pixel.y - sz / 2, sz, sz, func_data.color);
                subtr += val * mult;
                mult = -1;
            }
            /* if (last_subtr * subtr < 0 && functions.size() == 2) {
                glm::vec2 collide = point_to_pixel({x, val});
                pix.fill_circle(collide.x, collide.y, 5, func_data.color);
            } */
            last_subtr = subtr;
        }
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("microblock_fraction");
        return sq;
    }

    void mark_data_unchanged() override {}
    void change_data() override {} // RealFunctionScene has no DataObjects
    bool check_if_data_changed() const override { return false; }

    // On ending a transition, update each function if it was transitioning.
    void on_end_transition_extra_behavior(const TransitionType tt) override {
        // TODO we don't track transition type...
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

