#include "GeometryScene.h"
#include <vector>
#include <cmath>
#include <sstream>
#include <unordered_set>
#include <list>
#include <utility>
#include <algorithm>
#include "../../Host_Device_Shared/helpers.h"

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

    double gm = get_geom_mean_size();
    double line_thickness = gm/200.;
    int point_color = 0xffffffff;
    int line_color = 0xff6666ff;
    int text_color = 0xffffffff;
    float microblock_fraction = 0.5;
    if(state.contains("microblock_fraction_passthrough")) microblock_fraction = state["microblock_fraction_passthrough"];

    float bounce = 1 - square(square(microblock_fraction - 1));
    float interp = smoother2(microblock_fraction);

    Pixels geometry(pix.wh);

    for(const GeometricLine& l : construction.lines) {
        if(!l.draw_shape) continue;
        vec2 start_point = l.start;
        vec2 end_point = l.end;
        if(l.use_state) {
            start_point = vec2(state["line_"+l.identifier+"_start_x"], state["line_"+l.identifier+"_start_y"]);
            end_point = vec2(state["line_"+l.identifier+"_end_x"], state["line_"+l.identifier+"_end_y"]);
        }
        vec2 start_pixel = point_to_pixel(start_point);
        vec2 end_pixel = point_to_pixel(end_point);
        const vec2 mid_pixel = (start_pixel + end_pixel) / 2.f;
        if(!l.old) {
            // Multiply line length by bounce
            start_pixel = mid_pixel + (start_pixel - mid_pixel) * bounce;
            end_pixel = mid_pixel + (end_pixel - mid_pixel) * bounce;
        }
        geometry.bresenham(start_pixel.x, start_pixel.y, end_pixel.x, end_pixel.y, line_color, 1, line_thickness*.75);
    }
    for(const GeometricPoint& p : construction.points) {
        vec2 position = p.position;
        if(p.use_state) position = vec2(state["point_"+p.identifier+"_x"], state["point_"+p.identifier+"_y"]);
        const vec2 position_pixel = point_to_pixel(position);
        double radius = line_thickness * p.width_multiplier * 2;
        if(p.draw_shape){
            if(!p.old) {
                double radius_pop = line_thickness * p.width_multiplier * 8 * bounce;
                radius = min(radius, radius_pop);
                geometry.fill_circle(ivec2(position_pixel.x, position_pixel.y), radius_pop, point_color, (1-interp)*.8);
            }
            geometry.fill_circle(ivec2(position_pixel.x, position_pixel.y), radius, point_color, 1);
        }
        if(p.label != "" && p.width_multiplier > .4) {
            ScalingParams sp(vec2(160, 16) * line_thickness * p.width_multiplier);
            Pixels latex = latex_to_pix(latex_color(text_color, p.label), sp);
            geometry.overlay_cpu(latex, ivec2(position_pixel.x - latex.wh.x/2, position_pixel.y - line_thickness * 6 - latex.wh.y/2), p.old ? 1 : interp);
        }
    }

    pix.overlay_gpu(geometry, vec2(), construction_opacity);
}

const StateQuery GeometryScene::populate_state_query() const {
    StateQuery sq = CoordinateScene::populate_state_query();
    sq.insert("construction_opacity");
    for(const GeometricPoint& p : construction.points) {
        if(!p.old) {
            sq.insert("microblock_fraction_passthrough");
            break;
        }
    }
    for(const GeometricLine& l : construction.lines) {
        if(!l.old) {
            sq.insert("microblock_fraction_passthrough");
            break;
        }
    }
    for (const GeometricPoint& p : construction.points) {
        if(p.use_state) {
            sq.insert("point_"+p.identifier+"_x");
            sq.insert("point_"+p.identifier+"_y");
        }
    }
    for (const GeometricLine& l : construction.lines) {
        if(l.use_state) {
            sq.insert("line_"+l.identifier+"_start_x");
            sq.insert("line_"+l.identifier+"_start_y");
            sq.insert("line_"+l.identifier+"_end_x");
            sq.insert("line_"+l.identifier+"_end_y");
        }
    }
    return sq;
}

void GeometryScene::on_end_transition_extra_behavior(const TransitionType tt) {
    // TODO make this micro or macroblock based
    construction.set_all_old();
}
