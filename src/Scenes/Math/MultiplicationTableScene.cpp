
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

        {"show_0", "1"},
        {"show_1", "1"},
        {"show_2", "1"},
        {"show_3", "1"},
        {"show_4", "1"},
    });
}


void MultiplicationTableScene::draw() {

    vector<string> units = {"","1","i","j","ij"};
    vector<string> signs = {"","-"};
    vector<uint> unit_colors = {0x00000000,0x00cccccc,0x0033cccc,0x00cc33cc,0x00cccc33};
    
    vector<int> table_units = {
        0,1,2,3,4,
        1,1,2,3,4,
        2,2,1,4,3,
        3,3,4,1,2,
        4,4,3,2,1,
    };
    vector<int> table_signs;

    if (state["commutative"] == 1){
        table_signs = {
            0,0,0,0,0,
            0,0,0,0,0,
            0,0,1,0,1,
            0,0,0,1,1,
            0,0,1,1,0,
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

    ivec2 wh = get_width_height();
    const int square_size = min(wh.x,wh.y)*0.8;
    const ivec2 offset = (wh-ivec2(square_size,square_size))*0.5;
    const int cell_gap = square_size*0.01;
    const int cell_size = square_size*0.2;
    const ivec2 cell_radius = (ivec2(cell_size,cell_size)-cell_gap)/2;

    const uint bg_color = 0xff000000 | ((int) state["bg_1_r"]) << 16 | ((int) state["bg_1_g"]) << 8 | (int) state["bg_1_b"];

    for (int r = 0; r < 5; r++){
        for (int c = 0; c < 5; c++){
            int i = r*5 + c;
            float cell_opacity = min(state["show_" + to_string(c)],state["show_" + to_string(r)]);
            uint32_t cell_alpha = (int) (255*cell_opacity) << 24;

            ivec2 cell_center = offset + ivec2(cell_size*(r+0.5), cell_size*(c+0.5));
            if (r > 0 && c > 0){
                draw_rectangle(gpu_pix.get_ptr(), get_width_height(), 
                    cell_center-cell_radius, cell_center+cell_radius, 
                    cell_alpha + unit_colors[table_units[i]]
                );
            }

            if (table_signs[i] == 1){
                write_text(gpu_pix.get_ptr(), get_width_height(), 
                    latex_color(bg_color, signs[table_signs[i]]+units[table_units[i]]), 
                    cell_center, cell_radius*2, 1, 0
                );

            } else {
                draw_rectangle(gpu_pix.get_ptr(), get_width_height(), 
                    cell_center-cell_radius*0.9, cell_center+cell_radius*0.9, 
                    bg_color
                );
                write_text(gpu_pix.get_ptr(), get_width_height(), 
                    latex_color(unit_colors[table_units[i]], signs[table_signs[i]]+units[table_units[i]]), 
                    cell_center, cell_radius*2, cell_opacity, 0
                );
            }

        }
    }

}


