#include "BeaverGridSpacetimeScene.h"

extern "C" void beaver_grid_spacetime(
    uint32_t* pixels, ivec2 wh, vec2 lx_ty, vec2 rx_by,
    vec2 grid_wh, vec2 spacetime_wh, float tm_border, float iterations
);

BeaverGridSpacetimeScene::BeaverGridSpacetimeScene(const vec2& dimension)
: Scene(dimension) {
    manager.set({
        {"iterations", "1"},
        {"grid_w", "81"},
        {"grid_h", "81"},
        {"spacetime_w", "11"},
        {"spacetime_h", "11"},
        {"center_x", "0.5"},
        {"center_y", "0.5"},
	{"zoom", "0"}
    });
}

void BeaverGridSpacetimeScene::draw() {
    float scale = pow(2.718281828f, -state["zoom"]);
    ivec2 wh = get_width_height();
    vec2 grid_wh(state["grid_w"],state["grid_h"]);
    vec2 spacetime_wh(state["spacetime_w"],state["spacetime_h"]);
    vec2 lx_ty;
    vec2 rx_by;
    if (grid_wh.x / grid_wh.y < float(wh.x)/wh.y) {
        lx_ty = vec2(state["center_x"] - grid_wh.y * wh.x * scale / (2 * grid_wh.x * wh.y), state["center_y"] - scale / 2);
        rx_by = vec2(state["center_x"] + grid_wh.y * wh.x * scale / (2 * grid_wh.x * wh.y), state["center_y"] + scale / 2);
    } else {
        lx_ty = vec2(state["center_x"] - scale / 2, state["center_y"] - grid_wh.x * wh.y * scale / (2 * grid_wh.y * wh.x));
        rx_by = vec2(state["center_x"] + scale / 2, state["center_y"] + grid_wh.x * wh.y * scale / (2 * grid_wh.y * wh.x));
    }
    float tm_border = 0.1;
    beaver_grid_spacetime(
        gpu_pix.get_ptr(), get_width_height(), lx_ty, rx_by,
        grid_wh, spacetime_wh, tm_border, state["iterations"]
    );
}
