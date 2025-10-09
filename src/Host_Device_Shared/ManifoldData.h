// Simple header used by both manifold.cu and ManifoldScene.cpp
struct ManifoldData {
    const char* x_eq;
    const char* y_eq;
    const char* z_eq;
    const char* r_eq;
    const char* i_eq;
    float u_min;
    float u_max;
    int u_steps;
    float v_min;
    float v_max;
    int v_steps;
};
