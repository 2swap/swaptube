#include "PermutationScene.h"
#include "../../DataObjects/Permutation.h"
#include "../../Host_Device_Shared/helpers.h"
#include <cstdint>
#include <vector>
#include <cmath>

extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void cuda_draw_bezier(
    uint32_t* pix, const ivec2& wh, const vec2& p1, const vec2& p2, const vec2& p3, 
    const vec2& p4, const vec2& lx_ty, const vec2& rx_by);

void PermutationScene::on_end_transition_extra_behavior(const TransitionType tt) {
    if (moving_orbits.empty()) {
        return;
    }

    // Appliquer le décalage pour chaque orbite enregistrée
    for (const auto& [orbit_name, multiplier] : moving_orbits) {
        vector<string> moving_orbit = the_perm.orbits[orbit_name];
        
        for (int step = 0; step < multiplier; ++step) {
            uint32_t temp_color = the_perm.pieces[moving_orbit.back()];
            for (int i = (int)moving_orbit.size() - 2; i >= 0; i--) {
                the_perm.pieces[moving_orbit[i + 1]] = the_perm.pieces[moving_orbit[i]];
            }
            the_perm.pieces[moving_orbit[0]] = temp_color;
        }
    }

    // Réinitialiser la liste après la fin de la transition
    moving_orbits.clear();
}

void PermutationScene::move(const string orbit_name, int multiplier) {
    // Ajoute ou met à jour l'orbite dans la map
    moving_orbits[orbit_name] = multiplier;
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
    const float tension = 0.25f;

    // 1. Dessiner le squelette des orbites (courbes de Bézier)
    for (const auto& [orbit_name, orbit] : the_perm.orbits) {
        for (size_t i = 0; i < orbit.size(); i++) {
            const vec2& p1 = get_place_position_from_state(orbit[i]);
            const vec2& p2 = get_place_position_from_state(orbit[(i + 1) % orbit.size()]);
            const vec2& p3 = get_place_position_from_state(orbit[(i + 2) % orbit.size()]);
            const vec2& p4 = get_place_position_from_state(orbit[(i + 3) % orbit.size()]);
            const vec2& cp1 = p2 + (p3 - p1) * tension;
            const vec2& cp2 = p3 + (p2 - p4) * tension;
            cuda_draw_bezier(gpu_pix.get_ptr(), get_width_height(), p2, cp1, cp2, p3, 
                vec2(state["left_x"], state["top_y"]),
                vec2(state["right_x"], state["bottom_y"]));
        }
    }

    // 2. Dessiner chaque pièce
    for (const auto& [name, color] : the_perm.pieces) {
        bool is_moving = false;

        // Vérifier si cette pièce appartient à l'une des orbites en mouvement
        for (const auto& [orbit_name, multiplier] : moving_orbits) {
            const vector<string>& orbit = the_perm.orbits[orbit_name];
            
            int base_index = 0;
            for (const auto& piece_name : orbit) {
                if (piece_name == name) {
                    break;
                }
                base_index++;
            }

            // La pièce appartient à cette orbite !
            if (base_index < (int)orbit.size()) {
                is_moving = true;

                float total_progress = state["m"] * multiplier;
                int step_offset = static_cast<int>(std::floor(total_progress));
                float t = std::fmod(total_progress, 1.0f);

                int current_piece_index = (base_index + step_offset) % orbit.size();
                int N = orbit.size();

                vector<vec2> control_points = {
                    get_place_position_from_state(orbit[(current_piece_index - 1 + N) % N]),
                    get_place_position_from_state(orbit[current_piece_index]),
                    get_place_position_from_state(orbit[(current_piece_index + 1) % N]),
                    get_place_position_from_state(orbit[(current_piece_index + 2) % N])
                };

                vec2 cp1 = control_points[1] + (control_points[2] - control_points[0]) * tension;
                vec2 cp2 = control_points[2] + (control_points[1] - control_points[3]) * tension;

                vec2 center = point_to_pixel(bezier_2d(
                    control_points[1],
                    cp1,
                    cp2,
                    control_points[2],
                    t
                ));

                draw_circle(gpu_pix.get_ptr(), get_width_height(), center, 5.0f, color, 1.0f);
                break; // Une fois trouvée et dessinée, on peut sortir de la boucle des orbites
            }
        }

        // Si la pièce n'appartient à aucune orbite active en mouvement
        if (!is_moving) {
            const vec2& pos = point_to_pixel(get_place_position_from_state(name));
            draw_circle(gpu_pix.get_ptr(), get_width_height(), pos, 5.0f, color, 1.0f);
        }
    }
}