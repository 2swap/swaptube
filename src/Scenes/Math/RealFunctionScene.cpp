#include "RealFunctionScene.h"
#include <vector>
#include <stdexcept>
#include <string>

extern "C" void cuda_render_real_valued_function(
    uint32_t* d_pixels, const ivec2& wh,
    ResolvedStateEquationComponent* eq, int eq_size,
    const vec2& lx_ty, const vec2& rx_by
);

RealFunctionScene::RealFunctionScene(const vec2& dimensions) : CoordinateScene(dimensions) {
    manager.set({ {"function", "0"} });
}

void RealFunctionScene::draw() {
    vector<ResolvedStateEquationComponent> r = manager.get_resolved_equation("function");

    cuda_render_real_valued_function(
        gpu_pix->get_ptr(), get_width_height(),
        r.data(), r.size(),
        vec2(state["left_x"], state["top_y"]),
        vec2(state["right_x"], state["bottom_y"])
    );

    CoordinateScene::draw();
}

const StateQuery RealFunctionScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    state_query_insert_multiple(sq, {"function"});
    return sq;
}
