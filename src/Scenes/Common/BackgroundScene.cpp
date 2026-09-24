
#include "BackgroundScene.h"
#include "../../Host_Device_Shared/vec.h"

extern "C" void draw_rectangle(uint32_t* pix, const ivec2& wh, const ivec2& top_left, const ivec2& bottom_right, const uint32_t color);


extern "C" void background_render(
    const ivec2& wh,
    const ivec3 bg_0,
    const ivec3 bg_1,
    const float slider,
    unsigned int* d_colors
);

BackgroundScene::BackgroundScene(const vec2& dimensions) : Scene(dimensions){
    manager.set({
        {"bg_0_r", "0"},
        {"bg_0_g", "65"},
        {"bg_0_b", "65"},

        {"bg_1_r", "2"},
        {"bg_1_g", "65"},
        {"bg_1_b", "68"},

        {"slider", "0"},
    });
}


void BackgroundScene::draw() {

    const int opacity = 255 << 24;
        
    if (state["slider"] >= 1){
        const int bg_1 = ((int) state["bg_1_r"]) << 16 | ((int) state["bg_1_g"]) << 8 | (int) state["bg_1_b"];
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,0), get_width_height(), opacity + bg_1);

    } else if (state["slider"] <= 0){
        const int bg_0 = ((int) state["bg_0_r"]) << 16 | ((int) state["bg_0_g"]) << 8 | ((int) state["bg_0_b"]);
        draw_rectangle(gpu_pix.get_ptr(), get_width_height(), ivec2(0,0), get_width_height(), opacity + bg_0);

    } else {
        background_render(get_width_height(),
            ivec3((int) state["bg_0_r"], (int) state["bg_0_g"], (int) state["bg_0_b"]),
            ivec3((int) state["bg_1_r"], (int) state["bg_1_g"], (int) state["bg_1_b"]),
            state["slider"],
            gpu_pix.get_ptr()
        );
    }
    

}
