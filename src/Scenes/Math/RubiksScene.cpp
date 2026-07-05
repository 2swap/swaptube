#include "RubiksScene.h"
//#include <vector>
//#include "ManifoldScene.h"



extern "C" void cuda_render_cube(
    uint32_t* d_pixels, const ivec2& wh,
    float geom_mean_size,
    const quat& camera_direction, const vec3& camera_pos, float fov, float turn_fraction, quat rotation_quat, vec3 axis, float dist);

RubiksScene::RubiksScene(const vec2& dimensions) : ThreeDimensionScene(dimensions), rotation_quat(1, 0, 0, 0),
cut(vec3(0, 0, 0), 0) {
    manager.set({
        {"turn_fraction", "{microblock_fraction}"},
    });
    the_cube = new Rubiks(5); // cube created here
    add_data_object(the_cube);
}

quat get_quat_from_axis_angle(const vec3& axis, float angle) {
    float half_angle = angle / 2.0f;
    float sin_half_angle = sin(half_angle);
    return quat(cos(half_angle), axis.x * sin_half_angle, axis.y * sin_half_angle, axis.z * sin_half_angle);
}

void RubiksScene::exec_move_from_slice(const char move, const int depth) {
    cut = the_cube->cut_map[move][depth];
    rotation_quat = get_quat_from_axis_angle(cut.axis, 3.14159265358979323/2);
}


void RubiksScene::draw() {
    set_camera_direction();
    cuda_render_cube(gpu_pix->get_ptr(), get_width_height(), get_geom_mean_size(), camera_direction, 
    camera_pos, fov, smoother2(state["turn_fraction"]) , rotation_quat, cut.axis, cut.dist);
    //ThreeDimensionScene::draw();
}

const StateQuery RubiksScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"turn_fraction"});
    return s;
}



