#include "RubiksScene.h"
//#include <vector>
//#include "ManifoldScene.h"


extern "C" void cuda_render_cube(
    uint32_t* d_pixels, const ivec2& wh,
    float geom_mean_size,
    const quat& camera_direction, const vec3& camera_pos, float fov);

RubiksScene::RubiksScene(const vec2& dimensions) : ThreeDimensionScene(dimensions) {
    manager.set({
    });
}

void RubiksScene::draw() {
    set_camera_direction();
    cuda_render_cube(gpu_pix->get_ptr(), get_width_height(), get_geom_mean_size(), camera_direction, camera_pos, fov);
    //ThreeDimensionScene::draw();
}

const StateQuery RubiksScene::populate_state_query() const {
    StateQuery s = ThreeDimensionScene::populate_state_query();
    return s;
}



