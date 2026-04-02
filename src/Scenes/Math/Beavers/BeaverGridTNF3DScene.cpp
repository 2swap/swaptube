#include "BeaverGridTNF3DScene.h"

extern "C" void beaver_grid_TNF_3D_cuda(unsigned int* pixels, int w, int h, vec3 pos, quat camera, float fov, vec3 target, vec3 up, float use_quat_camera, float shell_border, float core_border, vec3 scale, int max_steps);

BeaverGridTNF3DScene::BeaverGridTNF3DScene(const vec2& dimension) {
    manager.set({
        {"fov", "1"},
        {"x", "0"},
        {"y", "0"},
        {"z", "0"},
        {"q1", "1"},
        {"qi", "0"},
        {"qj", "0"},
        {"qk", "0"},
        {"max_steps", "0"},
	{"target_x", "0"},
	{"target_y", "0"},
	{"target_z", "0"},
	{"up_x", "0"},
	{"up_y", "1"},
	{"up_z", "0"},
	{"use_quat_camera", "1"},
	{"shell_border", "0.1"},
	{"core_border", "0.1"},
	{"scale_x", "1"},
	{"scale_y", "1"},
	{"scale_z", "1"}
    });
}

void BeaverGridTNF3DScene::draw() {
    beaver_grid_TNF_3D_cuda(
        pix.pixels.data(), pix.w, pix.h,
        vec3(state["x"], state["y"], state["z"]),
        quat(state["q1"], state["qi"], state["qj"], state["qk"]), state["fov"],
	vec3(state["target_x"], state["target_y"], state["target_z"]),
	vec3(state["up_x"], state["up_y"], state["up_z"]),
        state["use_quat_camera"],
	state["shell_border"], state["core_border"],
	vec3(state["scale_x"], state["scale_y"], state["scale_z"]),
	state["max_steps"]
    );
}

const StateQuery BeaverGridTNF3DScene::populate_state_query() const {
    StateQuery sq = {"fov", "x", "y", "z", "q1", "qi", "qj", "qk", "max_steps", "target_x", "target_y", "target_z", "up_x", "up_y", "up_z", "use_quat_camera", "shell_border", "core_border", "scale_x", "scale_y", "scale_z"};
    return sq;
}

void BeaverGridTNF3DScene::mark_data_unchanged() { }
void BeaverGridTNF3DScene::change_data() { }
bool BeaverGridTNF3DScene::check_if_data_changed() const { return false; }
