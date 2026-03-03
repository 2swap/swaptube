#pragma once

#include "../Common/CoordinateScene.h"
#include "../../Core/State/StateManager.h"
#include <vector>
#include <complex>
#include <string>

class ComplexPlotScene : public CoordinateScene {
private:
    vector<complex<float>> last_computed_roots;
    int degree;

    void decrement_degree();
    void increment_degree();

public:
    ComplexPlotScene(const int d, const vec2& dimensions);

    void transition_root_rings(const TransitionType tt, const float opacity);
    void transition_coefficient_rings(const TransitionType tt, const float opacity);
    void transition_coefficient_opacities(const TransitionType tt, const float opacity);

    int get_degree() const;
    void set_degree(int d);

    void roots_to_coefficients();
    void coefficients_to_roots();

    vector<complex<float>> get_coefficients() const;
    void stage_swap(TransitionType tt, const string& root1, const string& root2, bool root_or_coefficient, bool clockwise = false);
    void reorder_roots(vector<complex<float>>& roots) const;
    vector<complex<float>> get_roots() const;

    void draw() override;

    const StateQuery populate_state_query() const override;
};
