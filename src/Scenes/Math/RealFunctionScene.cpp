#pragma once

#include "../Scene.cpp"
#include <vector>

struct FunctionData {
    std::string function;
    std::string transition_function;
    uint32_t color;

    FunctionData(const std::string& f, const std::string& tf, uint32_t c)
        : function(f), transition_function(tf), color(c) {}
};

class RealFunctionScene : public Scene {
public:
    RealFunctionScene(const double width = 1, const double height = 1)
        : Scene(width, height) {}

    std::pair<int, int> coordinate_to_pixel(std::pair<double, double> coordinate) {
        return std::make_pair( coordinate.first  * pixel_width * get_height() +  get_width() / 2.,
                              -coordinate.second * pixel_width * get_height() + get_height() / 2.);
    }

    double call_the_function(double x, const FunctionData& func_data) {
        double stf = state["microblock_fraction"];
        string xstring = to_string(x);
        string f1 = replace_substring(func_data.function, "?", xstring);
        string f2 = replace_substring(func_data.transition_function, "?", xstring);
        double y1 = calculator(f1);
        double y2 = f2.size() > 0 ? calculator(f2) : y1;
        return smoothlerp(y1, y2, stf);
    }

    void begin_transition(int index, const string& s){
        if (index < 0 || index >= functions.size()) throw runtime_error("Bad function index!");
        functions[index].transition_function = s;
    }

    void draw() override {
        render_axes();
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

    void render_dot(const pair<int, int>& pixel, uint32_t color) {
        uint32_t prevcol = pix.get_pixel(pixel.first, pixel.second);
        pix.set_pixel(pixel.first, pixel.second, color);
        pix.set_pixel(pixel.first, pixel.second + 1, color);
        pix.set_pixel(pixel.first, pixel.second - 1, color);
    }

    void render_point(const pair<int, int>& pixel) {
        pix.fill_ellipse(pixel.first, pixel.second, 10, 10, OPAQUE_WHITE);
    }

    void render_axes() {
        std::pair<int, int> i_pos = coordinate_to_pixel(make_pair(0, 10));
        std::pair<int, int> i_neg = coordinate_to_pixel(make_pair(0, -10));
        std::pair<int, int> r_pos = coordinate_to_pixel(make_pair(10, 0));
        std::pair<int, int> r_neg = coordinate_to_pixel(make_pair(-10, 0));
        pix.bresenham(i_pos.first, i_pos.second, i_neg.first, i_neg.second, 0xff004488, 1, 2);
        pix.bresenham(r_pos.first, r_pos.second, r_neg.first, r_neg.second, 0xff004488, 1, 2);
    }

    void render_functions() {
        double bound = get_width() / (2. * pixel_width * get_height());
        double dx = 10. / get_width();
        double last_subtr = 0;
        for (double x = -bound; x <= bound; x += dx) {
            double subtr = 0;
            int mult = 1;
            double val;
            for (const auto& func_data : functions) {
                val = call_the_function(x, func_data);
                const pair<double, double> point = make_pair(x, val);
                render_dot(coordinate_to_pixel(point), func_data.color);
                subtr += val * mult;
                mult = -1;
            }
            if(last_subtr * subtr < 0 && functions.size() == 2) {
                pair<int, int> collide = coordinate_to_pixel(make_pair(x, val));
                render_point(collide);
                pix.bresenham(collide.first, collide.second, collide.first, get_height()/2, 0xffffffff, 1, 2);
            }
            last_subtr = subtr;
        }
    }

    const StateQuery populate_state_query() const override {
        return StateQuery{"microblock_fraction"};
    }

    void mark_data_unchanged() override {}
    void change_data() override {} // ComplexPlotScene has no DataObjects
    bool check_if_data_changed() const override { return false; } // ComplexPlotScene has no DataObjects

    void on_end_transition(bool is_macroblock) {
        for(int i = 0; i < functions.size(); i++){
            if(functions[i].transition_function != "")
                functions[i].function = functions[i].transition_function;
            functions[i].transition_function = "";
        }
    }

private:
    double pixel_width = 0.1;
    vector<FunctionData> functions;
};

