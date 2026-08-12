#include "BeaverGridSpacetimeScene.h"

extern "C" void beaver_grid_spacetime(
    uint32_t* pixels, ivec2 wh, vec2 lx_ty, vec2 rx_by,
    ivec2 grid_wh, ivec2 spacetime_wh, float tm_border, float iterations
);

BeaverGridSpacetimeScene::BeaverGridSpacetimeScene(const vec2& dimension)
: Scene(dimension) {
    manager.set({
        {"iterations", "1"},
        {"center_x", "0.5"},
        {"center_y", "0.5"},
	{"zoom", "0"}
    });
}

void BeaverGridSpacetimeScene::draw() {
    float scale = pow(2.718281828f, -state["zoom"]);
    vec2 lx_ty(state["center_x"] - 8 * scale / 9, state["center_y"] - scale / 2);
    vec2 rx_by(state["center_x"] + 8 * scale / 9, state["center_y"] + scale / 2);
    ivec2 grid_wh(81,81);
    ivec2 spacetime_wh(11,11);
    float tm_border = 0.1;
    beaver_grid_spacetime(
        gpu_pix.get_ptr(), get_width_height(), lx_ty, rx_by,
        grid_wh, spacetime_wh, tm_border, state["iterations"]
    );
}
