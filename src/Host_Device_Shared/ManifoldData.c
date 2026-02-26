#pragma once

#include "../Core/State/ResolvedStateEquationComponent.c"
// Simple header used by both manifold.cu and ManifoldScene.cpp
struct ManifoldData {
    size_t x_size;
    ResolvedStateEquationComponent* x_eq;
    size_t y_size;
    ResolvedStateEquationComponent* y_eq;
    size_t z_size;
    ResolvedStateEquationComponent* z_eq;
    size_t r_size;
    ResolvedStateEquationComponent* r_eq;
    size_t i_size;
    ResolvedStateEquationComponent* i_eq;
    float u_min;
    float u_max;
    int u_steps;
    float v_min;
    float v_max;
    int v_steps;
};
