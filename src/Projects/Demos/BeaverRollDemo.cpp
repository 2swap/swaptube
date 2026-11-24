#include "../Scenes/Math/Beavers/BeaverTNF3DScene.h"
#include <string>

void render_video() {
    BeaverTNF3DScene bs;
    bs.manager.set({
        {"target_x", ".5"},
        {"target_y", ".5"},
        {"target_z", ".5"},
        {"q1", "1"},
        {"qi", ".5"},
        {"qj", ".5"},
        {"qk", "0"},
        {"axis_cycle_start", "10"},
        {"camera_distance", "1.5"},
        {"fov", ".5"},
        {"max_steps", "50"},
        {"max_tnf_depth", "15"},
    });

    stage_macroblock(SilenceBlock(10));
    for(int i = 9; i >= 0; i--) {
        bs.manager.transition(MICRO, "axis_cycle_start", std::to_string(i));
        bs.render_microblock();
    }
}
