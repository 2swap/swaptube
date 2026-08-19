#include "ThreeDAlgebraScene.h"

ThreeDAlgebraScene::ThreeDAlgebraScene(const vec2& dimensions) : ThreeDimensionScene(dimensions) {
    manager.set({
        {"associativity", "0"}, // 0 = left-associated (e*a)*b, 1 = right-associated e*(a*b)
        {"xx_x", "1"}, {"xx_y", "0"}, {"xx_z", "0"},
        {"xy_x", "0"}, {"xy_y", "1"}, {"xy_z", "0"},
        {"xz_x", "0"}, {"xz_y", "0"}, {"xz_z", "1"},
        {"yy_x", "-1"}, {"yy_y", "0"}, {"yy_z", "0"},
        {"yz_x", "-1"}, {"yz_y", "0"}, {"yz_z", "0"},
        {"zz_x", "0"}, {"zz_y", "1"}, {"zz_z", "0"},
        {"a_x", "0"}, {"a_y", "1"}, {"a_z", "0"},
        {"b_x", "0"}, {"b_y", "0"}, {"b_z", "1"},
        {"lines_thickness_multiplier", "2"},
    });
}

vec3 ThreeDAlgebraScene::multiply(const vec3& p, const vec3& q,
                                   const vec3& xx, const vec3& xy, const vec3& xz,
                                   const vec3& yy, const vec3& yz, const vec3& zz) const {
    return xx * (p.x * q.x)
         + xy * (p.x * q.y + p.y * q.x)
         + xz * (p.x * q.z + p.z * q.x)
         + yy * (p.y * q.y)
         + yz * (p.y * q.z + p.z * q.y)
         + zz * (p.z * q.z);
}

void ThreeDAlgebraScene::draw() {
    set_camera_direction();
    clear_lines();

    const vec3 xx(state["xx_x"], state["xx_y"], state["xx_z"]);
    const vec3 xy(state["xy_x"], state["xy_y"], state["xy_z"]);
    const vec3 xz(state["xz_x"], state["xz_y"], state["xz_z"]);
    const vec3 yy(state["yy_x"], state["yy_y"], state["yy_z"]);
    const vec3 yz(state["yz_x"], state["yz_y"], state["yz_z"]);
    const vec3 zz(state["zz_x"], state["zz_y"], state["zz_z"]);
    const vec3 a(state["a_x"], state["a_y"], state["a_z"]);
    const vec3 b(state["b_x"], state["b_y"], state["b_z"]);
    const float associativity = state["associativity"];
    const vec3 ab = multiply(a, b, xx, xy, xz, yy, yz, zz);

    const int reach = 2;
    const int width = 2 * reach + 1;
    vector<vec3> transformed(width * width * width);
    auto index = [&](int ix, int iy, int iz) {
        return ((ix + reach) * width + (iy + reach)) * width + (iz + reach);
    };
    for (int ix = -reach; ix <= reach; ix++)
        for (int iy = -reach; iy <= reach; iy++)
            for (int iz = -reach; iz <= reach; iz++) {
                const vec3 e(ix, iy, iz);
                const vec3 left  = multiply(multiply(e, a, xx, xy, xz, yy, yz, zz), b, xx, xy, xz, yy, yz, zz);
                const vec3 right = multiply(e, ab, xx, xy, xz, yy, yz, zz);
                transformed[index(ix, iy, iz)] = left * (1 - associativity) + right * associativity;
            }

    const uint32_t grid_color = 0xffffffff;
    for (int ix = -reach; ix <= reach; ix++)
        for (int iy = -reach; iy <= reach; iy++)
            for (int iz = -reach; iz <= reach; iz++) {
                const vec3& p = transformed[index(ix, iy, iz)];
                if (ix < reach) add_line(Line(p, transformed[index(ix + 1, iy, iz)], grid_color, 1, false));
                if (iy < reach) add_line(Line(p, transformed[index(ix, iy + 1, iz)], grid_color, 1, false));
                if (iz < reach) add_line(Line(p, transformed[index(ix, iy, iz + 1)], grid_color, 1, false));
            }

    ThreeDimensionScene::draw();
}
