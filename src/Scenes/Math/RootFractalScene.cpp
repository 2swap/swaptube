#pragma once

#include "../Common/CoordinateScene.cpp"
#include "ComplexPlotScene.cpp" // Contains definition of populate_roots
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

class RootFractalScene : public CoordinateScene {
public:
    RootFractalScene(const float width = 1, const float height = 1) : CoordinateScene(width, height) {
        state_manager.set("coefficient0_r", "0");
        state_manager.set("coefficient0_i", "0");
        state_manager.set("coefficient1_r", "1");
        state_manager.set("coefficient1_i", "0");
        state_manager.set("terms", "8");
        state_manager.set("floor_terms", "<terms> floor");
        state_manager.set("dot_radius", "1");
    }

    void draw() override {
        cout << "Drawing RootFractalScene" << flush;
        int terms = int(state["floor_terms"]);
        int iter_count = 1 << terms; // 2^terms
        complex<float> coeff0(state["coefficient0_r"], state["coefficient0_i"]);
        complex<float> coeff1(state["coefficient1_r"], state["coefficient1_i"]);
        float zoom = state["zoom"];
        float radius = zoom * get_geom_mean_size() * state["dot_radius"] / 20;

        for(int i = iter_count-1; i >= 0; i--){ // Reverse order to draw smaller polynomials on top
            vector<complex<float>> coefficients(terms);
            vector<complex<float>> roots;
            for(int bit = 0; bit < terms; bit++){
                coefficients[bit] = (i & (1 << bit)) ? coeff1 : coeff0;
            }
            while(coefficients.size() > 0){
                if(coefficients.back() == complex<float>(0,0)) coefficients.pop_back();
                else break;
            }
            if(coefficients.size() < 2) continue; // Constant polynomials don't have roots
            populate_roots(coefficients, roots);
            int size = floor(log2(i)) + 2;
            double rainbow = size/3.;
            int color = OKLABtoRGB(255, .8, lerp(-.233888, .276216, cos(rainbow)/2.), lerp(-.311528, .198570, sin(rainbow)/2.));
            for(const auto& root : roots){
                const glm::vec2 pixel(point_to_pixel(glm::vec2(root.real(), root.imag())));
                pix.fill_circle(pixel.x, pixel.y, 1 + radius / size, color);
            }
        }

        CoordinateScene::draw();
    }

    const StateQuery populate_state_query() const override {
        StateQuery sq = CoordinateScene::populate_state_query();
        sq.insert("coefficient0_r");
        sq.insert("coefficient0_i");
        sq.insert("coefficient1_r");
        sq.insert("coefficient1_i");
        sq.insert("floor_terms");
        sq.insert("dot_radius");
        sq.insert("zoom");
        return sq;
    }
};
