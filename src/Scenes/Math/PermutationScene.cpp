#include "PermutationScene.h"
#include "../../DataObjects/Permutation.h"
#include "../../Host_Device_Shared/helpers.h"
#include <cstdint>
#include <vector>
// #include <vector>

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color);
extern "C" void cuda_draw_bezier(
    uint32_t* pix, const ivec2& wh, const vec2& p1, const vec2& p2, const vec2& p3, 
    const vec2& p4, const vec2& lx_ty, const vec2& rx_by);


void PermutationScene::on_end_transition_extra_behavior(const TransitionType tt) {
    if (moving_orbit_name == "") {
        return;
    }
    vector<string> moving_orbit = the_perm.orbits[moving_orbit_name];
    uint32_t temp_color = the_perm.pieces[moving_orbit.back()];
    for (int i=moving_orbit.size()-2; i >= 0; i--) {
        the_perm.pieces[moving_orbit[(i+1)]] = the_perm.pieces[moving_orbit[i]];
    }
    the_perm.pieces[moving_orbit[0]] = temp_color;
    moving_orbit_name = "";
}

void PermutationScene::move(const string orbit_name) {
    moving_orbit_name = orbit_name;
}

PermutationScene::PermutationScene(const string file_name, const vec2& dimensions) : CoordinateScene(dimensions), the_perm(file_name) {
    manager.set({
        {"m", "{microblock_fraction}"},
    });
    for (const auto& [place_name, point] : the_perm.places) {
        manager.set({
            {place_name + ".x", to_string(point.x)},
            {place_name + ".y", to_string(point.y)},
        });
    }
}

vec2 PermutationScene::get_place_position_from_state(const string& place_name) {
    float x = state[place_name + ".x"];
    float y = state[place_name + ".y"];
    return vec2(x, y);
}

void PermutationScene::draw() {
    // Adjust this value to control the "tightness" of the curve
    const float tension = 0.25f;

    // draw every orbit here with a for each orbit in orbits
    for (const auto& [orbit_name, orbit] : the_perm.orbits) {
        for (int i=0; i < orbit.size(); i++) {
            const vec2& p1 = get_place_position_from_state(orbit[i]);
            const vec2& p2 = get_place_position_from_state(orbit[(i+1)%orbit.size()]);
            const vec2& p3 = get_place_position_from_state(orbit[(i+2)%orbit.size()]);
            const vec2& p4 = get_place_position_from_state(orbit[(i+3)%orbit.size()]);
            const vec2& cp1 = p2 + (p3 - p1) * tension;
            const vec2& cp2 = p3 + (p2 - p4) * tension;
            cuda_draw_bezier(gpu_pix.get_ptr(), get_width_height(), p2, cp1, cp2, p3, 
            vec2(state[ "left_x"], state[   "top_y"]),
            vec2(state["right_x"], state["bottom_y"]));
        }
    }

    // get the orbit name
    // for piece in pieces, draw_circle
    for (const auto& [name, color] : the_perm.pieces) {
        const vec2& pos = point_to_pixel(get_place_position_from_state(name));
        const vector<string> orbit = the_perm.orbits[moving_orbit_name];
        int current_piece_index = 0;
        for (const auto& piece_name : orbit) {
            if (piece_name == name) {
                break;
            }
            current_piece_index++;
        }
        if (current_piece_index >= orbit.size()) {
            draw_circle(gpu_pix.get_ptr(), get_width_height(), pos, 5.0f, color);
            continue;
        }
        // print the piece index
        cout << "Piece " << name << " is at index " << current_piece_index << " in orbit " << moving_orbit_name << endl;
        vector<vec2> control_points = {
        get_place_position_from_state(orbit[(current_piece_index + orbit.size() - 1)%orbit.size()]),
            get_place_position_from_state(orbit[current_piece_index]),
            get_place_position_from_state(orbit[(current_piece_index+1)%orbit.size()]),
            get_place_position_from_state(orbit[(current_piece_index+2)%orbit.size()])};
        cout << "Control points for piece " << name << ": ";
        for (const auto& cp : control_points) {
            cout << "(" << cp.x << ", " << cp.y << ") ";
        }
        cout << endl;

        

        // adjusting control points to make bezier curve go through the current piece's position
        vec2 cp1 = control_points[1] + (control_points[2] - control_points[0]) * tension;
        vec2 cp2 = control_points[2] + (control_points[1] - control_points[3]) * tension;

        vec2 center = point_to_pixel(bezier_2d(
            control_points[1],
            cp1,
            cp2,
            control_points[2],
            state["m"]));
        //cout << "Drawing piece " << name << " at pixel position " << center.x << ", " << center.y << " with color " << std::hex << color << std::dec << endl;
        draw_circle(gpu_pix.get_ptr(), get_width_height(), center, 5.0f, color);
    }
}
