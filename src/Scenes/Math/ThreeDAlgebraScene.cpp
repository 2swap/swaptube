#include "ThreeDAlgebraScene.h"

ThreeDAlgebraScene::ThreeDAlgebraScene(const vec2& dimensions) : ThreeDimensionScene(dimensions) {
    manager.set({
        {"associativity", "0"}, // 0 = left-associated (e*a)*b, 1 = right-associated e*(a*b)
        {"xx_x", "1"}, {"xx_y", "0"}, {"xx_z", "0"},
        {"xy_x", "0"}, {"xy_y", "1"}, {"xy_z", "0"},
        {"xz_x", "0"}, {"xz_y", "0"}, {"xz_z", "1"},
        {"yy_x", "-1"}, {"yy_y", "0"}, {"yy_z", "0"},
        {"yz_x", "-1"}, {"yz_y", "0"}, {"yz_z", "0"}, {"yz_w", "0"},
        {"zz_x", "0"}, {"zz_y", "1"}, {"zz_z", "0"}, {"zz_w", "0"},
        {"a_x", "0"}, {"a_y", "1"}, {"a_z", "0"}, {"a_w", "0"},
        {"b_x", "0"}, {"b_y", "0"}, {"b_z", "1"}, {"b_w", "0"},
        {"lines_thickness_multiplier", "2"},
        {"w_slider", "0"},
    });
}

vec4 ThreeDAlgebraScene::multiply(const vec4& p, const vec4& q,
                                   const vec4& xx, const vec4& xy, const vec4& xz,
                                   const vec4& yy, const vec4& yz, const vec4& zz) const {

    vec4 yzz = (p.z * q.w + p.w * q.z)*zz;

    return xx * (p.x * q.x)
         + xy * (p.x * q.y + p.y * q.x)
         + xz * (p.x * q.z + p.z * q.x)
         + yy * (p.y * q.y)
         + yz * (p.y * q.z + p.z * q.y + p.x * q.w + p.w * q.x)
         + zz * (p.z * q.z - p.w * q.w)
            + vec4(-yzz.y, yzz.x,- p.y * q.w - p.w * q.y -yzz.w, yzz.z);
}

vec3 ThreeDAlgebraScene::project(const vec4& p) const {
    return vec3(p.x+p.w*0.4, p.y+p.w*0.3, p.z+p.w*0.2);
}

void ThreeDAlgebraScene::draw() {
    set_camera_direction();
    clear_lines();

    const vec4 xx(state["xx_x"], state["xx_y"], state["xx_z"], 0);
    const vec4 xy(state["xy_x"], state["xy_y"], state["xy_z"], 0);
    const vec4 xz(state["xz_x"], state["xz_y"], state["xz_z"], 0);
    const vec4 yy(state["yy_x"], state["yy_y"], state["yy_z"], 0);
    const vec4 yz(state["yz_x"], state["yz_y"], state["yz_z"], state["yz_w"]);
    const vec4 zz(state["zz_x"], state["zz_y"], state["zz_z"], state["zz_w"]);
    const vec4 a(state["a_x"], state["a_y"], state["a_z"], state["a_w"]);
    const vec4 b(state["b_x"], state["b_y"], state["b_z"], state["b_w"]);
    const float associativity = state["associativity"];
    const vec4 ab = multiply(a, b, xx, xy, xz, yy, yz, zz);
    const float w = state["w_slider"];


    const int reach = 2;
    const int width = 2 * reach + 1;
    vector<vec3> transformed(width * width * width * 2);
    auto index = [&](int ix, int iy, int iz, int iw) {
        return ((iw*width + ix + reach) * width + (iy + reach)) * width + (iz + reach);
    };
    for (int ix = -reach; ix <= reach; ix++)
        for (int iy = -reach; iy <= reach; iy++)
            for (int iz = -reach; iz <= reach; iz++) {
                const vec4 e(ix, iy, iz, 0);
                const vec3 left  = project(multiply(multiply(e, a, xx, xy, xz, yy, yz, zz), b, xx, xy, xz, yy, yz, zz));
                const vec3 right = project(multiply(e, ab, xx, xy, xz, yy, yz, zz));
                transformed[index(ix, iy, iz, 0)] = left * (1 - associativity) + right * associativity;

                if (w > 0){
                    const vec4 ew(ix, iy, iz, w);
                    const vec3 leftw  = project(multiply(multiply(ew, a, xx, xy, xz, yy, yz, zz), b, xx, xy, xz, yy, yz, zz));
                    const vec3 rightw = project(multiply(ew, ab, xx, xy, xz, yy, yz, zz));
                    transformed[index(ix, iy, iz, 1)] = leftw * (1 - associativity) + rightw * associativity;
                }
            }

    const uint32_t grid_color = 0xffffffff;
    for (int ix = -reach; ix <= reach; ix++)
        for (int iy = -reach; iy <= reach; iy++)
            for (int iz = -reach; iz <= reach; iz++) {
                const vec3& p = transformed[index(ix, iy, iz, 0)];
                if (ix < reach) add_line(Line(p, transformed[index(ix + 1, iy, iz, 0)], grid_color, 1, false));
                if (iy < reach) add_line(Line(p, transformed[index(ix, iy + 1, iz, 0)], grid_color, 1, false));
                if (iz < reach) add_line(Line(p, transformed[index(ix, iy, iz + 1, 0)], grid_color, 1, false));


                if (w > 0){
                    const vec3& q = transformed[index(ix, iy, iz, 1)];
                    if (ix < reach) add_line(Line(q, transformed[index(ix + 1, iy, iz, 1)], grid_color, 1, false));
                    if (iy < reach) add_line(Line(q, transformed[index(ix, iy + 1, iz, 1)], grid_color, 1, false));
                    if (iz < reach) add_line(Line(q, transformed[index(ix, iy, iz + 1, 1)], grid_color, 1, false));
                    add_line(Line(q, p, grid_color, 1, false));
                }
            }

    ThreeDimensionScene::draw();
}
