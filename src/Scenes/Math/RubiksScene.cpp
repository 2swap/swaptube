#include "RubiksScene.h"
//#include <vector>
//#include "ManifoldScene.h"

extern "C" void allocate_stickers(char (*d_stickers)[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE], int num_stickers);

extern "C" void copy_stickers(char d_stickers[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE], char h_stickers[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE], int num_stickers);

extern "C" void cuda_render_cube(
    uint32_t* d_pixels, const ivec2& wh,
    float geom_mean_size,
    const quat& camera_direction, const vec3& camera_pos, float fov, float turn_fraction, quat rotation_quat, vec3 axis, float dist, char (*d_stickers)[6][MAX_CUBE_SIZE][MAX_CUBE_SIZE], int cube_size);

void RubiksScene::on_end_transition_extra_behavior(const TransitionType tt) {

    copy_stickers(d_stickers, the_cube->pattern.pattern, 6 * MAX_CUBE_SIZE * MAX_CUBE_SIZE);
    //cut = Cut(vec3(0, 0, 0), 0);
    rotation_quat = quat(1, 0, 0, 0);
}



RubiksScene::RubiksScene(const vec2& dimensions) : ThreeDimensionScene(dimensions), rotation_quat(1, 0, 0, 0),cut(vec3(0, 0, 0), 0) {
    manager.set({
        {"turn_fraction", "{microblock_fraction}"},
        {"cube_size", "3"},
        {"d", "9"},
        {"qi", "-0.25"},
        {"qj", "0.25"},
        {"fov", "2"},
    });
    the_cube = new Rubiks; // cube created here
    add_data_object(the_cube);
    allocate_stickers(&d_stickers, 6 * MAX_CUBE_SIZE * MAX_CUBE_SIZE);
    copy_stickers(d_stickers, the_cube->pattern.pattern, 6 * MAX_CUBE_SIZE * MAX_CUBE_SIZE);
}

RubiksScene::RubiksScene(const CubeStickerPattern& pattern, const vec2& dimensions) : ThreeDimensionScene(dimensions), rotation_quat(1, 0, 0, 0),cut(vec3(0, 0, 0), 0) {
    manager.set({
        {"turn_fraction", "{microblock_fraction}"},
        {"cube_size", "3"},
        {"d", "9"},
        {"qi", "-0.25"},
        {"qj", "0.25"},
        {"fov", "2"},
    });
    the_cube = new Rubiks(pattern); // cube created here
    add_data_object(the_cube);
    allocate_stickers(&d_stickers, 6 * MAX_CUBE_SIZE * MAX_CUBE_SIZE);
    copy_stickers(d_stickers, the_cube->pattern.pattern, 6 * MAX_CUBE_SIZE * MAX_CUBE_SIZE);
}

quat get_quat_from_axis_angle(const vec3& axis, float angle) {
    float half_angle = angle / 2.0f;
    float sin_half_angle = sin(half_angle);
    return quat(cos(half_angle), axis.x * sin_half_angle, axis.y * sin_half_angle, axis.z * sin_half_angle);
}

void RubiksScene::exec_move_from_slice(const std::string& token) {
    the_cube->exec(token);
    Move m = the_cube->parseMove(token);
    float size = state["cube_size"];
    float distance = -1.0f + (2.0f * static_cast<float>(size - m.depth)) / static_cast<float>(size);
    if(m.face == 'U') cut = Cut(vec3(0,  1,  0), distance);
    if(m.face == 'D') cut = Cut(vec3(0, -1,  0), distance);
    if(m.face == 'F') cut = Cut(vec3(0,  0, -1), distance);
    if(m.face == 'B') cut = Cut(vec3(0,  0,  1), distance);
    if(m.face == 'L') cut = Cut(vec3(-1, 0,  0), distance);
    if(m.face == 'R') cut = Cut(vec3(1,  0,  0), distance);
    switch (m.turns) {
        case 1:
            rotation_quat = get_quat_from_axis_angle(cut.axis, 3.14159265358979323/2);
            break;
        case 2:
            rotation_quat = get_quat_from_axis_angle(cut.axis, 3.14159265358979323);
            break;
        case 3:
            rotation_quat = get_quat_from_axis_angle(cut.axis, -3.14159265358979323/2);
            break;
        default:
            rotation_quat = quat(1, 0, 0, 0); // No rotation for invalid turns
            break;
    }
}


void RubiksScene::draw() {
    set_camera_direction();
    cuda_render_cube(gpu_pix->get_ptr(), get_width_height(), get_geom_mean_size(), camera_direction, 
    camera_pos, fov, smoother2(state["turn_fraction"]) , rotation_quat, cut.axis, cut.dist, &d_stickers, state["cube_size"]);
    //ThreeDimensionScene::draw();
}

const StateQuery RubiksScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    state_query_insert_multiple(s, {"turn_fraction", "cube_size"});
    return s;
}



