#include "GeometryScene.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../Host_Device_Shared/helpers.h"
#include "../../IO/Latex.h"

using std::vector;
using std::min;
using std::max;

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void cuda_render_lines_from_host(uint32_t* d_pixels, const ivec2& wh,
                                            const vec2* h_endpoints, int segment_count,
                                            const vec2& lx_ty, const vec2& rx_by,
                                            uint32_t color, float opacity, float thickness);

GeometryScene::GeometryScene(const vec2& dimensions)
    : CoordinateScene(dimensions) {
    manager.set({
        {"construction_opacity", "1"},
    });
}

void GeometryScene::draw() {
    CoordinateScene::draw();

    if(construction.size() == 0) return;
    const float construction_opacity = state["construction_opacity"];
    if(construction_opacity < 0.01) return;

    const double gm = get_geom_mean_size();
    const double line_thickness = gm/200.;
    const uint32_t point_color = 0xffffffff;
    const uint32_t line_color  = 0xffffffff;
    const uint32_t text_color  = 0xffffffff;

    float microblock_fraction = 0.5;
    if(state.contains("microblock_fraction_passthrough")) microblock_fraction = state["microblock_fraction_passthrough"];

    const float interp = smoother2(microblock_fraction);

    const ivec2 wh = get_width_height();
    const vec2 lx_ty(state["left_x"],  state["top_y"]);
    const vec2 rx_by(state["right_x"], state["bottom_y"]);

    const float thickness = max(1.0, line_thickness * 0.75);
    vector<vec2> live_segments, dying_segments;
    for(const GeometricLine& l : construction.lines) {
        if(!l.draw_shape) continue;
        vec2 start_point = l.start;
        vec2 end_point = l.end;
        if(l.use_state) {
            start_point = vec2(state["line_"+l.identifier+"_start_x"], state["line_"+l.identifier+"_start_y"]);
            end_point   = vec2(state["line_"+l.identifier+"_end_x"],   state["line_"+l.identifier+"_end_y"]);
        }
        if(l.dying) {
            dying_segments.push_back(start_point);
            dying_segments.push_back(end_point);
        } else {
            if(!l.old) {
                end_point = start_point + (end_point - start_point) * interp;
            }
            live_segments.push_back(start_point);
            live_segments.push_back(end_point);
        }
    }
    if(!live_segments.empty()) {
        cuda_render_lines_from_host(gpu_pix.get_ptr(), wh,
                                   live_segments.data(), live_segments.size() / 2,
                                   lx_ty, rx_by, line_color, construction_opacity, thickness);
    }
    if(!dying_segments.empty()) {
        cuda_render_lines_from_host(gpu_pix.get_ptr(), wh,
                                   dying_segments.data(), dying_segments.size() / 2,
                                   lx_ty, rx_by, line_color, construction_opacity * (1 - interp), thickness);
    }

    for(const GeometricPoint& p : construction.points) {
        vec2 position = p.position;
        if(p.use_state) position = vec2(state["point_"+p.identifier+"_x"], state["point_"+p.identifier+"_y"]);
        const vec2 position_pixel = point_to_pixel(position);
        const bool entering = !p.old && !p.dying;
        double radius = line_thickness * p.width_multiplier * 2;
        float dot_opacity = construction_opacity;
        float label_life = p.old ? 1.f : interp;
        if(p.dying) {
            radius      *= (1 - interp);   // shrink the dot away
            dot_opacity *= (1 - interp);   // ...and fade it
            label_life   = (1 - interp);
        }
        if(p.draw_shape){
            if(entering) {
                const double radius_pop = line_thickness * p.width_multiplier * 8 * interp;
                radius = min(radius, radius_pop);
                draw_circle(gpu_pix.get_ptr(), wh, position_pixel, radius_pop, point_color, (1-interp)*.8*construction_opacity);
            }
            draw_circle(gpu_pix.get_ptr(), wh, position_pixel, radius, point_color, dot_opacity);
        }
        if(p.label != "" && p.width_multiplier > .4) {
            const vec2 envelope = vec2(160, 16) * line_thickness * p.width_multiplier;
            const vec2 label_center(position_pixel.x, position_pixel.y - line_thickness * 6 - envelope.y * 0.5f);
            write_text(gpu_pix.get_ptr(), wh, latex_color(text_color, p.label), label_center, envelope, label_life * construction_opacity, 0);
        }
    }
}

void GeometryScene::on_end_transition_extra_behavior(const TransitionType tt) {
    // TODO make this micro or macroblock based
    construction.prune_dead();   // items that just finished fading out are gone now
    construction.set_all_old();
}
