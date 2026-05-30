#include "MandelbrotScene.h"
#include "../../Host_Device_Shared/vec.h"
#include <complex>

using std::complex;

extern "C" void mandelbrot_render(
    const ivec2& wh,
    const vec2& lx_ty,
    const vec2& rx_by,
    const complex<float>& seed_z, const complex<float>& seed_x, const complex<float>& seed_c,
    const vec3& pixel_parameter_multipliers,
    int max_iterations,
    float gradation,
    float phase_shift,
    unsigned int internal_color,
    unsigned int* d_colors
);

MandelbrotScene::MandelbrotScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({
        {"max_iterations", "200"},
        {"seed_z_r", "0"},
        {"seed_z_i", "0"},
        {"seed_x_r", "2"},
        {"seed_x_i", "0"},
        {"seed_c_r", "0"},
        {"seed_c_i", "0"},
        {"center_x", "<seed_z_r> <pixel_param_z> * <seed_x_r> <pixel_param_x> * <seed_c_r> <pixel_param_c> * + +"},
        {"center_y", "<seed_z_i> <pixel_param_z> * <seed_x_i> <pixel_param_x> * <seed_c_i> <pixel_param_c> * + +"},
        {"pixel_param_z", "0"}, // Julia set
        {"pixel_param_x", "0"}, // X set
        {"pixel_param_c", "1"}, // Mandelbrot set
        {"gradation", "1"},
        {"phase_shift", "{t}"},
    });
}

const StateQuery MandelbrotScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {"max_iterations", "seed_z_r", "seed_z_i", "seed_x_r", "seed_x_i", "seed_c_r", "seed_c_i", "pixel_param_z", "pixel_param_x", "pixel_param_c", "gradation", "phase_shift"});
    return sq;
}

void MandelbrotScene::draw() {
    vec3 pixel_params = normalize(vec3(state["pixel_param_z"], state["pixel_param_x"], state["pixel_param_c"]));
    complex<float> seed_z(state["seed_z_r"], state["seed_z_i"]);
    complex<float> seed_x(state["seed_x_r"], state["seed_x_i"]);
    complex<float> seed_c(state["seed_c_r"], state["seed_c_i"]);
    mandelbrot_render(get_width_height(),
                      vec2(state["left_x"], state["top_y"]),
                      vec2(state["right_x"], state["bottom_y"]),
                      seed_z, seed_x, seed_c,
                      pixel_params,
                      state["max_iterations"],
                      state["gradation"],
                      state["phase_shift"],
                      OPAQUE_BLACK,
                      gpu_pix->get_ptr()
    );
    CoordinateScene::draw();
}
