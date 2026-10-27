
#include "MultiplicationTableScene.h"
#include "../../Host_Device_Shared/vec.h"
#include "../../Host_Device_Shared/Color.h"
#include "../../Core/Pixels.h"
#include "../../IO/Latex.h"
#include <vector>
#include <stdexcept>
#include <string>


extern "C" void draw_circle(uint32_t* pix, const ivec2& wh, const vec2& center, const float radius, const uint32_t color, const float opacity);
extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);


MultiplicationTableScene::MultiplicationTableScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"commutative", "1"},
        {"bg_1_r", "0"},
        {"bg_1_g", "0"},
        {"bg_1_b", "68"},

        {"opacity_00", "1"},{"opacity_01", "1"},{"opacity_02", "1"},{"opacity_03", "1"},{"opacity_04", "1"},
        {"opacity_10", "1"},{"opacity_11", "1"},{"opacity_12", "1"},{"opacity_13", "1"},{"opacity_14", "1"},
        {"opacity_20", "1"},{"opacity_21", "1"},{"opacity_22", "1"},{"opacity_23", "1"},{"opacity_24", "1"},
        {"opacity_30", "1"},{"opacity_31", "1"},{"opacity_32", "1"},{"opacity_33", "1"},{"opacity_34", "1"},
        {"opacity_40", "1"},{"opacity_41", "1"},{"opacity_42", "1"},{"opacity_43", "1"},{"opacity_44", "1"},

        {"q_1j", "0"},{"q_ij", "0"},{"q_jj", "0"},
        {"triangle", "1"},
        {"table_size", "4"},

        {"tilted", "0"},{"i_flippy", "0"},{"three_d", "0"},
    });
}


void MultiplicationTableScene::draw() {

    vector<int> table_signs;
    vector<int> table_units;
    vector<string> units;
    vector<float> unit_q; // crossfade weight from the unit's label to a question mark
    vector<uint> unit_colors; 
    vector<string> signs = {"","-"};


    if (state["three_d"] == 0){
        const string iunit = (state["tilted"]==0)?"i":"\\frac{2i+2j+ij}{3}";
        units = {"","1",iunit,"j","ij","ij"};
        unit_q = {0,0,0,0,state["q_ij"],state["q_jj"]};
        // empty, white, pink, green, blue, blue
        unit_colors = {0x00000000,0x00e3e5fc,0x00d865ff,0x0032ed96,0x0000c7ff,0x0000c7ff};
        
        table_units = {
            0,1,2,3,4,
            1,1,2,3,4,
            2,2,1,4,3,
            3,3,5,1,2,
            4,4,3,2,1,
        };

        if (state["commutative"] == 1){
            table_signs = {
                0,0,0,0,0,
                0,0,0,0,0,
                0,0,1,0,1,
                0,0,0,1,1,
                0,0,1,1,0,
            };
        } else if (state["i_flippy"] == 1) {
            table_signs = {
                0,0,0,0,0,
                0,0,0,0,0,
                0,0,0,0,1,
                0,0,1,1,0,
                0,0,0,1,1,
            };
        } else {
            table_signs = {
                0,0,0,0,0,
                0,0,0,0,0,
                0,0,1,0,1,
                0,0,1,1,0,
                0,0,0,1,1,
            };
        }

    } else {

        units = {"","1","i","j","j","ij","j^2","-1"};
        unit_q = {0,0,0,0,state["q_1j"],state["q_ij"],state["q_jj"],0};
        // empty, white, pink, green, green, blue, white, yellow
        unit_colors = {0x00000000,0x00e3e5fc,0x00d865ff,0x0032ed96,0x0032ed96,0x0000c7ff,0x00e3e5fc,0x00fff96e};
        
        table_units = {
            0,1,2,3,0,
            1,1,2,4,0,
            2,2,7,5,0,
            3,0,0,6,0,
            0,0,0,0,0,
        };
        table_signs = {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,0,0,0,
        };
    }

    ivec2 wh = get_width_height();
    const float table_limit = state["table_size"]+1;
    const int cell_size = min(wh.x,wh.y)*0.16;
    const int cell_gap = cell_size*0.05;
    const int square_size = cell_size*table_limit;
    const ivec2 offset = (wh-ivec2(square_size,square_size))*0.5;
    const ivec2 cell_radius = (ivec2(cell_size,cell_size)-cell_gap)/2;

    const uint bg_color = 0xff000000 | ((int) state["bg_1_r"]) << 16 | ((int) state["bg_1_g"]) << 8 | (int) state["bg_1_b"];

    for (int r = 0; r < table_limit; r++){
        for (int c = 0; c < table_limit; c++){

            float cell_opacity = state["opacity_" + to_string(r) + to_string(c)]*min(1.0f,table_limit-max(r,c));

            if (c < r && c != 0){
                if (state["triangle"]==1.0){
                    continue;
                } else {
                    cell_opacity *= (1.0-state["triangle"]);
                }
            }

            if (cell_opacity == 0) continue;

            int i = r*5 + c;

            uint32_t cell_alpha = ((int) (255*cell_opacity)) << 24;

            ivec2 cell_center = offset + ivec2(cell_size*(c+0.5), cell_size*(r+0.5));
            if (r > 0 && c > 0){
                draw_rectangle(gpu_pix.get_ptr(), get_width_height(), 
                    cell_center-cell_radius, cell_center+cell_radius, 
                    cell_alpha + unit_colors[table_units[i]]
                );
            }

            const string label = signs[table_signs[i]]+units[table_units[i]];
            const float q = unit_q[table_units[i]];
            auto write_label = [&](uint32_t color, float opacity){
                if (q < 1) write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(color, label), cell_center, cell_radius*2, opacity*(1-q), 0);
                if (q > 0) write_text(gpu_pix.get_ptr(), get_width_height(), latex_color(color, "?"), cell_center, cell_radius*2, opacity*q, 0);
            };

            if (table_signs[i] == 1){
                write_label(bg_color, 1);

            } else {
                if (r > 0 && c > 0){
                    draw_rectangle(gpu_pix.get_ptr(), get_width_height(), 
                        cell_center-cell_radius*0.9, cell_center+cell_radius*0.9, 
                        bg_color
                    );
                }
                write_label(unit_colors[table_units[i]], cell_opacity);
            }

        }
    }

}


