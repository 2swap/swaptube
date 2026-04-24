#include <vector>

#include "GeographyScene.h"

extern "C" void cuda_render_sphere(
    uint32_t* h_pixels, int width, int height,
    float geom_mean_size,
    const vec3& center, float radius,
    uint32_t* d_map, int map_width, int map_height,
    const quat& camera_direction, const vec3& camera_pos, float fov);
extern "C" uint32_t* cuda_copy_map(const string& filename_with_or_without_suffix, int& out_width, int& out_height);
extern "C" void cuda_free_map(uint32_t* d_map);

//void png_to_raw_data(uint32_t*& unallocated_data, int& width, int& height, const string& filename_with_or_without_suffix) {
GeographyScene::GeographyScene(const vec2& dimensions) : ThreeDimensionScene(dimensions) {
    uint32_t* ptr;
    int w, h;
    cuda_copy_map("earth_small", w, h);
}

void GeographyScene::draw() {
    ThreeDimensionScene::draw();
    cuda_render_sphere(
        pix.pixels.data(), pix.w, pix.h,
        1.0f,
        vec3(0.0f, 0.0f, 0.0f), 1.0f,
        d_map, 1024, 512,
        camera_direction, camera_pos, fov
    );
}

GeographyScene::~GeographyScene() {
    cuda_free_map(d_map);
}
